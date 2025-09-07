# ARGO Core Concepts

- **Surface Temperature**: Refers to temperature measurements taken at a shallow depth, where the pressure is less than 10 dbar. When a user asks for "surface temperature" or "sea surface temperature", filter the `levels` table using `WHERE levels.pres_dbar < 10`.

- **Surface Salinity**: Refers to salinity measurements taken at a shallow depth, where the pressure is less than 10 dbar. When a user asks for "surface salinity", filter the `levels` table using `WHERE levels.pres_dbar < 10`.

- **Surface Pressure**: Refers to the shallowest pressure measurement available for a profile. To find the average surface pressure, you should find the minimum pressure for each profile and then average those minimums.