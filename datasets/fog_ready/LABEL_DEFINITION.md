# Fog Label Definition

Date: 2026-05-05

This note documents how labels are derived from the KNMI hourly station dataset in `era5_merged.csv`.

## Source fields

- `VV_station`: KNMI coded horizontal visibility class
- `W1_station`: hourly fog indicator
- `WW_station`: WMO 4680 present weather code aggregated over the preceding 6 hours

## Time convention

KNMI documents the hourly timestamps as UTC and states that the timestamp indicates the end of the measuring interval preceding that timestamp.

Daily reference aggregations in this repo are currently computed by UTC calendar day.

## Visibility decoding

The KNMI `VV` field is a coded visibility scale, not direct metres.

The experiment-prep script preserves:

- `visibility_code`
- `visibility_lower_m`
- `visibility_upper_m`
- `visibility_m`

Where:

- `visibility_code` is the raw KNMI code
- `visibility_lower_m` and `visibility_upper_m` are decoded visibility-bin bounds
- `visibility_m` is the approximate midpoint of the decoded bin

For open-ended high-visibility code `89`, `visibility_m` is set to the lower bound.

## Fog label

`fog_label` is defined as:

1. `1` if `W1_station == 1`
2. `0` if `W1_station == 0`
3. otherwise fallback to visibility:
   - `1` if `VV_station <= 9`
   - `0` if `VV_station > 9`
4. missing if both `W1_station` and `VV_station` are missing

This makes the explicit KNMI fog indicator the primary source and uses low visibility as a fallback when the indicator is unavailable.

## Low-visibility label

`low_visibility_label` is defined from `VV_station` only:

- `1` if `VV_station <= 49`
- `0` if `VV_station >= 50`
- missing if `VV_station` is missing

This corresponds to visibility below `5 km`.

## Visibility class

`visibility_class` is defined from `VV_station` as:

- `3`: `VV_station <= 1`  -> below `200 m`
- `2`: `VV_station in [2, 9]` -> `200 m` to below `1000 m`
- `1`: `VV_station in [10, 49]` -> `1000 m` to below `5000 m`
- `0`: `VV_station >= 50` -> `5000 m` and above

## Present weather code

`present_weather_code` is copied from `WW_station`.

Additionally:

- `fog_from_weather_code = 1` when `present_weather_code` is in `[40, 49]`

This field is kept as supporting context and not used as the primary fog label source.

## Pressure semantics

Important distinction:

- `P_station` is KNMI sea-level pressure and is stored as `msl_pressure_obs`
- `era5_sp` is ERA5 surface pressure and is stored as `surface_pressure`

These are not the same physical variable and should not be treated as interchangeable.
