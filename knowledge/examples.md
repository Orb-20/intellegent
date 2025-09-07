# Complex Query Examples

- **Question**: "Compare the average temperature in the Arabian Sea and the Bay of Bengal."
- **SQL**:
SELECT
  CASE
    WHEN (latitude BETWEEN 8 AND 25 AND longitude BETWEEN 50 AND 75) THEN 'Arabian Sea'
    WHEN (latitude BETWEEN 5 AND 22 AND longitude BETWEEN 80 AND 100) THEN 'Bay of Bengal'
  END as sea_name,
  AVG(levels.temp_degc) as average_temperature
FROM profiles
JOIN levels ON profiles.profile_id = levels.profile_id
WHERE
  (latitude BETWEEN 8 AND 25 AND longitude BETWEEN 50 AND 75) OR
  (latitude BETWEEN 5 AND 22 AND longitude BETWEEN 80 AND 100)
GROUP BY sea_name;