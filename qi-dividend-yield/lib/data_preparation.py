import duckdb
from pathlib import Path

def index_data_prep(data_dir, outputs_dir):
    """
    Purpose:     
        Organize the sp500 dataset and calculate dividend yields
    
    Inputs:
        data_dir       folder with input CSV files
        outputs_dir    folder with output CSV files
    
    Outputs:
        outputs/sp500_monthly.csv
    """
    query = f"""
        WITH organized AS (
            SELECT
                CAST(date_trunc('month', MthCalDt) AS DATE) AS Month,
                MthTotRet AS Total_ret,
                MthPrcRet AS Price_ret,
                MthPrcInd AS Price_raw,
                MthIncRet AS Income_ret,
                INDNO AS Index_id,
                IndNm AS Index_name,
                MthCalDt AS Date
            FROM read_csv('{data_dir}/sp500_universe_monthly_raw.csv')
        ), prices AS (
            SELECT
                *, 
                PRODUCT(
                    CASE
                        WHEN Month = DATE '1925-12-01' THEN 1
                        ELSE 1 + Price_ret
                    END                
                ) OVER (
                    ORDER BY Month
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS Price
            FROM organized
        ), monthly AS (
            SELECT
                *,
                LAG(Price) OVER (ORDER BY Month) * (Total_ret - Price_ret) 
                AS dividend
            FROM prices
        ), annual AS (
            SELECT
                *,
                CASE
                    WHEN COUNT(dividend) OVER (
                        ORDER BY Month
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                    ) = 12
                    THEN SUM(dividend) OVER (
                        ORDER BY Month
                        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                    )
                    ELSE NULL
                END AS dividend_12m
            FROM monthly                    
        )
        SELECT
            *,
            dividend_12m / Price AS yield_end,
            dividend_12m / LAG(Price, 12) OVER (ORDER BY Month)
            AS yield_start
        FROM annual
        ORDER BY Month;    
    """
    sp500 = duckdb.sql(query)
    sp500.write_csv(str(outputs_dir / "sp500_monthly.csv"))

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
                Ticker,
                CAST(date_trunc('month', MthCalDt) AS DATE) AS Month,
                MthRet,
                MthPrc
            FROM read_csv('{data_dir}/stock_monthly_returns.csv')
        ), flags AS (
            SELECT DISTINCT
                PERMNO,
                date_trunc('month', MthCalDt) AS Month,
                MthRetFlg,
                SecurityEndDt
            FROM read_csv('{data_dir}/stock_monthly_raw.csv')
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
            JOIN read_csv('{data_dir}/stock_otc_prices_raw.csv') AS o
                ON c.Symbol = o.symbol
                AND c.Exit_date = o.closingbestbiddate
        ), adjusted AS (
            SELECT
                s.PERMNO,
                s.Ticker,
                s.Month,
                CASE
                    WHEN f.MthRetFlg = 'IP'
                        THEN (1 + s.MthRet) * o.Price / s.MthPrc - 1
                    ELSE s.MthRet
                END AS Stock_ret,
                f.MthRetFlg AS Return_flag,
                CASE
                    WHEN f.MthRetFlg = 'DE' OR o.PERMNO IS NOT NULL
                        THEN TRUE
                    ELSE FALSE
                END AS Exit_flag,
                CASE
                    WHEN o.PERMNO IS NOT NULL THEN o.Exit_date
                    WHEN f.MthRetFlg = 'DE' THEN f.SecurityEndDt
                    ELSE NULL
                END AS Exit_date
            FROM stocks AS s
            LEFT JOIN flags AS f
                ON s.PERMNO = f.PERMNO AND s.Month = f.Month
            LEFT JOIN otc AS o
                ON s.PERMNO = o.PERMNO AND s.Month = o.Month
        ), market AS (
            SELECT
                Month,
                Total_ret AS Market_ret
            FROM read_csv('{outputs_dir}/sp500_monthly.csv')
        ), rf AS (
            SELECT
                date_trunc('month', MthCalDt) AS Month,
                MthTotRet AS Rf
            FROM read_csv('{data_dir}/rf_monthly_raw.csv')
        ), combined AS (
            SELECT
                s.*,
                m.Market_ret,
                r.Rf
            FROM adjusted AS s
            LEFT JOIN market AS m ON s.Month = m.Month
            LEFT JOIN rf AS r ON s.Month = r.Month
        )
        SELECT
            *,
            Stock_ret - Rf AS stock_excess,
            Market_ret - Rf AS market_excess
        FROM combined
        ORDER BY PERMNO, Month;
    """
    stocks = duckdb.sql(query)
    stocks.write_csv(str(outputs_dir / "stock_monthly.csv"))


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    index_data_prep(data_dir, outputs_dir)
    stock_data_prep(data_dir, outputs_dir)

if __name__ == "__main__":
    main()
