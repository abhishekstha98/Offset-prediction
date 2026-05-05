# Data Requirements for Fog / Low-Visibility Forecasting

Date: 2026-05-02

## Objective

Prepare a station-level **hourly** dataset for fog / low-visibility prediction.

The final unit of data should be:

`one row = one station x one valid hour`

Each row should align:

- station observations
- ERA5 / ERA5-Land / NWP predictors
- optional satellite predictors
- fog / visibility labels
- station metadata
- quality flags

The main modeling target is **fog / low visibility**, not daily temperature offset correction.

## Primary target labels

Direct visibility is preferred over proxy labels.

Please create at least these label columns:

- `fog_label`: `1` if `visibility_m <= 1000`, else `0`
- `low_visibility_label`: `1` if `visibility_m <= 5000`, else `0`
- `visibility_class`: ordinal class, for example:
  - `0`: `> 5000 m`
  - `1`: `1000 - 5000 m`
  - `2`: `200 - 1000 m`
  - `3`: `<= 200 m`

If present-weather codes are available, also include:

- `present_weather_code`
- `fog_reported`
- `mist_reported`

If visibility is unavailable at some stations, keep those rows with missing labels rather than inventing proxy labels.

## Required station metadata

Create:

`station_metadata.csv`

Columns:

- `station`
- `name`
- `lat`
- `lon`
- `height`
- `country`
- `station_type`
- `start_time_utc`
- `end_time_utc`

Notes:

- `station` must be a stable unique identifier.
- `time` fields must be in UTC.
- `station_type` should distinguish sources such as `airport`, `synop`, `aws`, or similar.

## Required hourly station observations

Create:

`station_obs_hourly.parquet`

One row per station-hour.

Minimum columns:

- `station`
- `time`
- `visibility_m`
- `present_weather_code`
- `t2m_obs`
- `td2m_obs`
- `rh_obs`
- `wind_speed_obs`
- `wind_dir_obs`
- `u10_obs`
- `v10_obs`
- `surface_pressure_obs`
- `precip_obs`

Strongly preferred if available:

- `cloud_base_obs`
- `cloud_cover_obs`
- `dew_formed_flag`
- `fog_reported`
- `mist_reported`

Also include QA fields where possible:

- `qa_flag_visibility`
- `qa_flag_temperature`
- `qa_flag_dewpoint`
- `qa_flag_wind`
- `qa_flag_pressure`

If a variable is unavailable, keep the column missing or omit it entirely, but do not fabricate values.

## Required ERA5 / ERA5-Land / model forcing

Create:

`era5_hourly_collocated.parquet`

One row per station-hour, already collocated to station coordinates.

Minimum columns:

- `station`
- `time`
- `era5_t2m`
- `era5_d2m`
- `era5_u10`
- `era5_v10`
- `era5_ws10`
- `surface_pressure`

Preferred additional columns:

- `mx2t`
- `mn2t`
- `blh`
- `tcc`
- `ssrd`
- `strd`
- `tp`
- `soil_moisture`
- `soil_temperature`

The current repo can derive these internally if the raw fields are present:

- `dewpoint_spread_2m`
- `rh_2m`
- `wind_speed_10m`
- `theta_v_2m`
- `theta_v_delta_1d`
- `t2m_delta_1d`
- `dewpoint_spread_delta_1d`

So there is no need to precompute them unless convenient.

## Optional satellite predictors

Satellite inputs are optional for v1, but useful if available.

Create:

`satellite_hourly_collocated.parquet`

One row per station-hour.

Recommended columns:

- `station`
- `time`
- `sat_source`
- `pixel_distance_km`
- `fog_low_stratus_prob`
- `cloud_mask`
- `low_cloud_mask`
- `cloud_top_temp`
- `cloud_top_height`
- `brightness_temp_10_8um`
- `brightness_temp_12_0um`
- `brightness_temp_3_9um`
- `bt_diff_10_8_12_0`
- `bt_diff_3_9_10_8`
- `land_surface_temp`
- `sat_qa_flag`

Notes:

- If satellite data are lower-frequency or not exactly hourly, align to the nearest valid prior observation and keep the original timestamp if possible.
- If only daily land-surface products are available, store them separately rather than pretending they are hourly.

## Alignment rules

All tables must follow these rules:

- use **UTC only**
- use exact hourly timestamps
- keep `station` and `time` as the join keys
- do not use future information
- for valid time `t`, predictors must come from `t` or earlier only

If multiple observations occur within the same hour:

- use a deterministic aggregation rule
- document it in the label note

Examples:

- nearest valid observation to top-of-hour
- hourly mean for continuous variables
- hourly minimum for visibility if operationally justified

## Missing-data policy

Follow this exactly:

- keep missing values as missing
- do not fill fog labels heuristically
- do not interpolate fog events
- do not backfill predictors from the future
- keep QA flags

Missing targets are acceptable.
The training pipeline can mask them.

## Pollution-specific data

Do **not** prioritize pollution-specific data for this version.

For the fog / low-visibility paper, you can ignore:

- emissions inventories
- PM2.5 / PM10 labels
- chemistry transport outputs
- traffic or industrial source maps

Those belong to a separate pollution forecasting task.

## Preferred merged training table

Provide a final merged file:

`fog_training_merged_hourly.parquet`

Minimum required columns:

- `station`
- `time`
- `lat`
- `lon`
- `height`
- `visibility_m`
- `fog_label`
- `low_visibility_label`
- `visibility_class`
- `era5_t2m`
- `era5_d2m`
- `era5_u10`
- `era5_v10`
- `era5_ws10`
- `surface_pressure`

Strongly preferred additional columns:

- `present_weather_code`
- `blh`
- `tcc`
- `ssrd`
- `strd`
- `tp`
- `cloud_base_obs`
- `cloud_cover_obs`
- `fog_low_stratus_prob`
- `brightness_temp_10_8um`
- `brightness_temp_12_0um`
- `brightness_temp_3_9um`
- `bt_diff_10_8_12_0`
- `bt_diff_3_9_10_8`

## Required summary report from data preparation

Please also provide:

`label_definition.md`

It should document:

- exact visibility thresholds used
- how `fog_label` was created
- how `low_visibility_label` was created
- how `visibility_class` was created
- how hourly aggregation was handled
- how duplicate observations were handled
- any station exclusions
- any known quality issues

Also include a short summary report with:

- number of stations
- time range
- sampling frequency
- percent missing for major columns
- number of fog hours
- number of low-visibility hours
- label counts by station
- label counts by month
- class imbalance summary
- number of rows with both station labels and satellite coverage

## One-line brief

Prepare an hourly station-level fog dataset with direct visibility labels, collocated ERA5 predictors, optional collocated satellite fog/low-cloud predictors, station metadata, QA flags, and no future leakage. The final table must be one row per station-hour in UTC.
