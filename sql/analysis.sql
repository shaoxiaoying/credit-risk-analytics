-- Run against data/processed/credit_clients_clean.csv using DuckDB.
-- Example: SELECT * FROM read_csv_auto('data/processed/credit_clients_clean.csv');

-- 1) Default rate by most recent repayment status.
SELECT pay_0 AS latest_repayment_status,
       COUNT(*) AS customers,
       AVG(default_payment_next_month) AS observed_default_rate
FROM read_csv_auto('data/processed/credit_clients_clean.csv')
GROUP BY 1
ORDER BY 1;

-- 2) Default rate by credit limit band.
SELECT CASE
         WHEN limit_bal < 50000 THEN '< NT$50k'
         WHEN limit_bal < 100000 THEN 'NT$50k–99k'
         WHEN limit_bal < 200000 THEN 'NT$100k–199k'
         WHEN limit_bal < 500000 THEN 'NT$200k–499k'
         ELSE 'NT$500k+'
       END AS credit_limit_band,
       COUNT(*) AS customers,
       AVG(default_payment_next_month) AS observed_default_rate
FROM read_csv_auto('data/processed/credit_clients_clean.csv')
GROUP BY 1
ORDER BY MIN(limit_bal);

-- 3) Default rate by number of recent months with a positive repayment delay.
SELECT CASE
         WHEN delayed_months = 0 THEN '0 delayed months'
         WHEN delayed_months = 1 THEN '1 delayed month'
         ELSE '2+ delayed months'
       END AS recent_delay_group,
       COUNT(*) AS customers,
       AVG(default_payment_next_month) AS observed_default_rate
FROM (
  SELECT *,
         (CASE WHEN pay_0 > 0 THEN 1 ELSE 0 END +
          CASE WHEN pay_2 > 0 THEN 1 ELSE 0 END +
          CASE WHEN pay_3 > 0 THEN 1 ELSE 0 END) AS delayed_months
  FROM read_csv_auto('data/processed/credit_clients_clean.csv')
)
GROUP BY 1
ORDER BY MIN(delayed_months);
