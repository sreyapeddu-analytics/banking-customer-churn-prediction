-- Banking customer churn: segment analysis queries
-- SQLite-compatible. Run against data/customers.csv imported as table `customers`.
-- Table schema: customer_id, age, tenure_months, balance, num_products,
--               has_credit_card, is_active_member, estimated_salary,
--               geography, gender, churned (1 = churned, 0 = retained)

-- 1. Churn rate by tenure band ---------------------------------------------
SELECT
    CASE
        WHEN tenure_months <= 12 THEN '0-12 months'
        WHEN tenure_months <= 36 THEN '13-36 months'
        WHEN tenure_months <= 60 THEN '37-60 months'
        ELSE '61+ months'
    END AS tenure_band,
    COUNT(*) AS customers,
    SUM(churned) AS churned_customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(100.0 * (COUNT(*) - SUM(churned)) / COUNT(*), 2) AS retention_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY churn_rate_pct DESC;

-- 2. Churn rate by number of products held ----------------------------------
SELECT
    num_products,
    COUNT(*) AS customers,
    SUM(churned) AS churned_customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 2) AS pct_of_customers
FROM customers
GROUP BY num_products
ORDER BY num_products;

-- 3. Churn rate by geography ------------------------------------------------
SELECT
    geography,
    COUNT(*) AS customers,
    SUM(churned) AS churned_customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY geography
ORDER BY churn_rate_pct DESC;

-- 4. Average balance: churned vs retained -----------------------------------
SELECT
    CASE WHEN churned = 1 THEN 'churned' ELSE 'retained' END AS segment,
    COUNT(*) AS customers,
    ROUND(AVG(balance), 2) AS avg_balance,
    ROUND(AVG(tenure_months), 1) AS avg_tenure_months,
    ROUND(AVG(num_products), 2) AS avg_num_products,
    ROUND(100.0 * SUM(is_active_member) / COUNT(*), 2) AS active_member_pct
FROM customers
GROUP BY churned
ORDER BY segment;
