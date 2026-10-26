import duckdb
from pathlib import Path

EXCLUDED_PERMNOS = (21413, 12093)

def stock_data_prep(data_dir, outputs_dir):
    """
    Purpose:
        Organize stock returns, adjust OTC exits and calculate excess returns

    Inputs:
        data_dir       folder with input CSV files
        outputs_dir    folder with output CSV files

    Outputs:
        outputs/stock_monthly.csv
    """
    query = f"""
        WITH stocks AS (
            SELECT
                PERMNO,
                MIN(Ticker) AS Ticker,
                CAST(date_trunc('month', MthCalDt) AS DATE) AS Month,
                MthRet,
                MthPrc,
                MthCap,
                MthPrcVol,
                MthPrcVolMissCnt,
                MthRetFlg,
                SecurityEndDt
            FROM read_csv('{data_dir}/crsp_monthly_2011_2025.csv')
            WHERE PERMNO NOT IN {EXCLUDED_PERMNOS}
            -- Collapse ticker-only duplicates; keep conflicting numeric rows for checks.
            GROUP BY ALL
        ), cases (PERMNO, Symbol, Exit_date) AS (
            VALUES
                (89244, 'WGHTQ', DATE '2025-05-16'),
                (91907, 'SMCI', DATE '2018-08-23')
        ), otc AS (
            SELECT
                c.PERMNO,
                date_trunc('month', c.Exit_date) AS Month,
                c.Exit_date,
                o.lastprice AS Price
            FROM cases AS c
            JOIN read_csv('{data_dir}/legacy_stock_otc_prices_raw.csv') AS o
                ON c.Symbol = o.symbol
                AND c.Exit_date = o.closingbestbiddate
        ), adjusted AS (
            SELECT
                s.PERMNO,
                s.Ticker,
                s.Month,
                s.MthRet AS Raw_stock_ret,
                s.MthPrc,
                s.MthCap,
                s.MthPrcVol,
                s.MthPrcVolMissCnt,
                CASE
                    WHEN s.MthRetFlg = 'IP'
                        THEN (1 + s.MthRet) * o.Price / s.MthPrc - 1
                    ELSE s.MthRet
                END AS Stock_ret,
                s.MthRetFlg AS Return_flag,
                CASE
                    WHEN s.MthRetFlg = 'DE' OR o.PERMNO IS NOT NULL
                        THEN TRUE
                    ELSE FALSE
                END AS Exit_flag,
                CASE
                    WHEN o.PERMNO IS NOT NULL THEN o.Exit_date
                    WHEN s.MthRetFlg = 'DE' THEN s.SecurityEndDt
                    ELSE NULL
                END AS Exit_date
            FROM stocks AS s
            LEFT JOIN otc AS o
                ON s.PERMNO = o.PERMNO AND s.Month = o.Month
        ), market AS (
            SELECT
                CAST(date_trunc('month', dateff) AS DATE) AS Month,
                mktrf AS market_excess,
                rf AS Rf
            FROM read_csv('{data_dir}/ff_monthly_2011_2025.csv')
        ), combined AS (
            SELECT
                s.*,
                m.market_excess,
                m.Rf
            FROM adjusted AS s
            LEFT JOIN market AS m ON s.Month = m.Month
        )
        SELECT
            *,
            Stock_ret - Rf AS stock_excess
        FROM combined
        ORDER BY PERMNO, Month;
    """
    stocks = duckdb.sql(query)
    stocks.write_csv(str(outputs_dir / "stock_monthly.csv"))

def market_features_prep(data_dir, outputs_dir, min_days = 15):
    """
    Purpose:
        Calculate market features and join volatility and financial ratios

    Inputs:
        data_dir       folder with input CSV files
        outputs_dir    folder with output CSV files
        min_days       minimum valid daily returns for volatility

    Outputs:
        outputs/features_monthly.csv
    """
    volatility_prep(data_dir, outputs_dir, min_days)
    financial_features_prep(data_dir, outputs_dir)
    query = f"""
        SELECT 
            s.PERMNO, s.Month,
            s.Stock_ret AS Recent_return,
            CASE
                WHEN s.MthCap > 0 THEN LN(s.MthCap)
                ELSE NULL
            END AS Size,
            CASE
                WHEN s.MthPrcVol > 0  AND s.MthPrcVolMissCnt = 0 
                    THEN LN(s.MthPrcVol)
                ELSE NULL
            END AS Trading_activity,
            CASE
                WHEN COUNT(s.Stock_ret) OVER(
                    PARTITION BY s.PERMNO
                    ORDER BY s.Month
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) = 5
                    THEN
                        PRODUCT(1 + s.Stock_ret) OVER (
                            PARTITION BY s.PERMNO
                            ORDER BY s.Month
                            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                        ) - 1 
                ELSE NULL
            END AS Momentum_6m,
            CASE
                WHEN COUNT(s.Stock_ret) OVER(
                    PARTITION BY s.PERMNO
                    ORDER BY s.Month
                    ROWS BETWEEN 11 PRECEDING AND 1 PRECEDING
                ) = 11
                    THEN
                        PRODUCT(1 + s.Stock_ret) OVER (
                            PARTITION BY s.PERMNO
                            ORDER BY s.Month
                            ROWS BETWEEN 11 PRECEDING AND 1 PRECEDING
                        ) - 1 
                ELSE NULL
            END AS Momentum_12m,
            v.Volatility,
            v.Valid_days,
            CASE
                WHEN REGR_COUNT(s.stock_excess, s.market_excess) OVER beta_window = 60
                    AND REGR_SXX(s.stock_excess, s.market_excess) OVER beta_window > 0
                    THEN REGR_SLOPE(s.stock_excess, s.market_excess) OVER beta_window
                ELSE NULL
            END AS beta,
            f.bm,
            f.GProf,
            f.debt_at,
            f.accrual,
            f.divyield
        FROM read_csv('{outputs_dir}/stock_monthly.csv') AS s
        LEFT JOIN read_csv(
            '{outputs_dir}/volatility_monthly.csv'
        )AS v
            ON s.PERMNO = v.PERMNO
            AND s.Month = v.Month
        LEFT JOIN read_csv(
            '{outputs_dir}/financial_features_monthly.csv'
        ) AS f
            ON s.PERMNO = f.PERMNO
            AND s.Month = f.Month
        -- Use 60 calendar months, including the current month.
        WINDOW beta_window AS (
            PARTITION BY s.PERMNO
            ORDER BY s.Month
            RANGE BETWEEN INTERVAL '59' MONTH PRECEDING AND CURRENT ROW
        )
        ORDER BY s.PERMNO, s.Month;
    """
    features = duckdb.sql(query)
    features.write_csv(str(outputs_dir / "features_monthly.csv"))


def volatility_prep(data_dir, outputs_dir, min_days=15):
    """
    Purpose:
        Calculate daily return volatility for each stock month

    Inputs:
        data_dir       folder with input CSV files
        outputs_dir    folder with output CSV files
        min_days       minimum number of valid daily returns

    Outputs:
        outputs/volatility_monthly.csv
    """
    if not isinstance(min_days, int) or min_days < 2:
        raise ValueError('min_days must be an integer of at least 2')

    duplicate = duckdb.sql(f"""
        SELECT PERMNO, DlyCalDt
        FROM read_csv('{data_dir}/crsp_daily_2016_2025_reused.csv')
        WHERE PERMNO NOT IN {EXCLUDED_PERMNOS}
        GROUP BY PERMNO, DlyCalDt
        HAVING COUNT(*) > 1
        LIMIT 1;
    """).fetchone()
    if duplicate is not None:
        raise ValueError('Duplicate PERMNO/day rows; check daily data')

    query = f"""
        WITH daily AS (
            SELECT
                PERMNO,
                CAST(date_trunc('month', DlyCalDt) AS DATE) AS Month,
                DlyRet
            FROM read_csv('{data_dir}/crsp_daily_2016_2025_reused.csv')
            WHERE PERMNO NOT IN {EXCLUDED_PERMNOS}
        )
        SELECT
            PERMNO,
            Month,
            COUNT(DlyRet) AS Valid_days,
            CASE
                WHEN COUNT(DlyRet) >= {min_days} THEN STDDEV_SAMP(DlyRet)
                ELSE NULL
            END AS Volatility
        FROM daily
        GROUP BY PERMNO, Month
        ORDER BY PERMNO, Month;
    """
    volatility = duckdb.sql(query)
    volatility.write_csv(str(outputs_dir / "volatility_monthly.csv"))


def financial_features_prep(data_dir, outputs_dir):
    """
    Purpose:
        Organize WRDS financial ratios by stock and month

    Inputs:
        data_dir       folder with input CSV files
        outputs_dir    folder with output CSV files

    Outputs:
        outputs/financial_features_monthly.csv
    """
    query = f"""
        SELECT DISTINCT
            permno AS PERMNO,
            CAST(date_trunc('month', public_date) AS DATE) AS Month,
            bm,
            GProf,
            debt_at,
            accrual,
            divyield
        FROM read_csv('{data_dir}/wrds_financial_ratios_numeric_2015_2025.csv')
        WHERE permno NOT IN {EXCLUDED_PERMNOS}
        ORDER BY PERMNO, Month;
    """
    financial = duckdb.sql(query)
    duplicate = financial.query('ratios', """
        SELECT PERMNO, Month
        FROM ratios
        GROUP BY PERMNO, Month
        HAVING COUNT(*) > 1
        LIMIT 1;
    """).fetchone()
    if duplicate is not None:
        raise ValueError('Conflicting financial ratios for a PERMNO/month')
    financial.write_csv(str(outputs_dir / "financial_features_monthly.csv"))


def model_features_prep(outputs_dir):
    query = f"""
        WITH eligible AS (
            SELECT *
            FROM read_csv('{outputs_dir}/features_monthly.csv')
            WHERE Recent_return IS NOT NULL
                AND Size IS NOT NULL
                AND Trading_activity IS NOT NULL
                AND Momentum_6m IS NOT NULL
                AND Momentum_12m IS NOT NULL
                AND Volatility IS NOT NULL
                AND beta IS NOT NULL
                AND COALESCE(bm, GProf, debt_at, accrual, divyield) IS NOT NULL
        )
        SELECT
            PERMNO,
            Month,
            Recent_return,
            Size,
            Trading_activity,
            Momentum_6m,
            Momentum_12m,
            Volatility,
            Valid_days,
            beta,
            COALESCE(bm, MEDIAN(bm) OVER monthly) AS bm,
            COALESCE(GProf, MEDIAN(GProf) OVER monthly) AS GProf,
            COALESCE(debt_at, MEDIAN(debt_at) OVER monthly) AS debt_at,
            COALESCE(accrual, MEDIAN(accrual) OVER monthly) AS accrual,
            COALESCE(divyield, MEDIAN(divyield) OVER monthly) AS divyield
        FROM eligible
        WINDOW monthly AS (PARTITION BY Month)
        ORDER BY PERMNO, Month;
    """
    features = duckdb.sql(query)
    missing = features.query('filled', """
        SELECT Month
        FROM filled
        WHERE bm IS NULL OR GProf IS NULL OR debt_at IS NULL
            OR accrual IS NULL OR divyield IS NULL
        LIMIT 1;
    """).fetchone()
    if missing is not None:
        raise ValueError(f'No monthly median available for a financial feature in {missing[0]}')
    features.write_csv(str(outputs_dir / "model_features_monthly.csv"))


def rank_features_prep(outputs_dir):
    feature_names = [
        'Recent_return', 'Size', 'Trading_activity',
        'Momentum_6m', 'Momentum_12m', 'Volatility', 'beta',
        'bm', 'GProf', 'debt_at', 'accrual', 'divyield'
    ]
    rank_columns = []
    for name in feature_names:
        rank_columns.append(f"""
            CASE
                WHEN COUNT(*) OVER monthly = 1 THEN 0.0
                ELSE 2.0 * (
                    RANK() OVER (PARTITION BY Month ORDER BY {name})
                    + (COUNT(*) OVER (PARTITION BY Month, {name}) - 1) / 2.0
                    - 1
                ) / (COUNT(*) OVER monthly - 1) - 1
            END AS {name}
        """)
    rank_columns = ','.join(rank_columns)
    query = f"""
        SELECT * REPLACE ({rank_columns})
        FROM read_csv('{outputs_dir}/model_features_monthly.csv')
        WINDOW monthly AS (PARTITION BY Month)
        ORDER BY PERMNO, Month;
    """
    features = duckdb.sql(query)
    features.write_csv(str(outputs_dir / "ranked_features_monthly.csv"))


def model_dataset_prep(outputs_dir):
    query = f"""
        SELECT
            f.*,
            CAST(f.Month + INTERVAL '1' MONTH AS DATE) AS Target_month,
            s.stock_excess AS Target_excess_return
        FROM read_csv('{outputs_dir}/ranked_features_monthly.csv') AS f
        LEFT JOIN read_csv('{outputs_dir}/stock_monthly.csv') AS s
            ON f.PERMNO = s.PERMNO
            AND f.Month + INTERVAL '1' MONTH = s.Month
        WHERE s.stock_excess IS NOT NULL
        ORDER BY f.PERMNO, f.Month;
    """
    dataset = duckdb.sql(query)
    duplicate = dataset.query('combined', """
        SELECT PERMNO, Month
        FROM combined
        GROUP BY PERMNO, Month
        HAVING COUNT(*) > 1
        LIMIT 1;
    """).fetchone()
    if duplicate is not None:
        raise ValueError('Duplicate PERMNO/month rows in model dataset')
    dataset.write_csv(str(outputs_dir / "model_dataset_monthly.csv"))


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    stock_data_prep(data_dir, outputs_dir)
    market_features_prep(data_dir, outputs_dir)
    model_features_prep(outputs_dir)
    rank_features_prep(outputs_dir)
    model_dataset_prep(outputs_dir)
