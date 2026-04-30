# Fog Forecasting Conversation Notes

Date: 2026-04-25

## Purpose

This note preserves the fog-forecasting research discussion for the Nepal-focused offset-prediction project. It summarizes fog types in Nepal, pollution and airport-visibility impacts, temperature inversion, ERA5-Land relevance, and the most important ERA5-Land-derived variables for a future fog-formation training setup.

## Fog Formations In Nepal

Nepal can experience several fog or fog-like low-cloud regimes because of its strong elevation gradients, winter inversions, river valleys, monsoon moisture, and the influence of the Indo-Gangetic Plain.

Main fog formations:

- **Radiation fog**: Common in winter mornings in the Terai and enclosed valleys. It forms when the ground cools rapidly overnight under clear skies and light winds, cooling near-surface air to saturation.
- **Valley fog**: Common in Kathmandu Valley, Pokhara Valley, Surkhet, Dang, and other hill valleys. Cold dense air drains downslope at night and pools on the valley floor.
- **Advection fog**: Can occur when moist air from the Indo-Gangetic Plain or local moist sources moves over cooler ground, especially in the Terai and foothills.
- **Upslope fog**: Forms when moist air is forced uphill and cools. It is relevant along foothills, ridges, and hill-airport approach paths.
- **Evaporation or steam fog**: Localized near rivers, wetlands, lakes, and irrigated land when cold air passes over warmer wet surfaces.
- **Frontal fog**: Occasional, usually linked with rain and cool near-surface air during winter disturbances.
- **Monsoon hill fog / low cloud**: Common in the mid-hills and mountain slopes during the monsoon. This is often meteorologically low cloud, but it behaves like fog at the surface.

The most common Nepal-relevant types for winter surface impacts are **radiation fog** and **valley fog**. For hills and mountains, **upslope fog** and **monsoon low cloud/fog** are also important.

## Pollution Impacts

Fog and pollution interact strongly when the atmosphere is stable and vertical mixing is suppressed.

Most pollution-relevant fog settings in Nepal:

- **Radiation fog**: Important in the Terai and Kathmandu Valley during winter mornings. The same calm, cold, stable conditions that support fog also trap PM2.5, PM10, vehicle emissions, brick-kiln smoke, biomass-burning pollution, and dust near the ground.
- **Valley fog**: Especially important in Kathmandu, Pokhara, Surkhet, Dang, and other enclosed valleys. Cold-air pooling creates or strengthens a temperature inversion, limiting vertical mixing.
- **Advection fog / moist haze**: Relevant in the Terai when moist polluted air from the Indo-Gangetic Plain moves northward and combines with local emissions.
- **Evaporation fog**: Usually local, but extra moisture near rivers, wetlands, and irrigation can increase particle growth and worsen haze.

For pollution applications, the key atmospheric setup is:

```text
cold surface air
+ weak wind
+ stable/inversion layer
+ high humidity
+ local or transported emissions
= trapped haze, smog, fog, and poor air quality
```

## Airport Visibility Impacts

Fog and low cloud reduce runway visibility, approach visibility, and terrain clearance. The dominant fog type depends on airport setting.

- **Terai airports**: Radiation fog and advection fog are most important. Relevant airports include Biratnagar, Bhairahawa, Nepalgunj, Janakpur, Simara, and Dhangadhi.
- **Valley airports**: Valley fog and radiation fog are most important. Relevant airports include Kathmandu and Pokhara.
- **Hill and mountain airports**: Upslope fog and monsoon low cloud are most important. These can obscure ridges, valleys, and approach corridors even when runway-level conditions vary rapidly.

For airport-visibility modeling, ERA5-Land variables are useful as predictors but are not enough for operational runway visibility by themselves. Airport observations, METAR/visibility records, local station data, and satellite fog/low-cloud products are needed for validation.

## Temperature Inversion

A temperature inversion is a reversal of the normal vertical temperature pattern.

Normally:

```text
higher altitude: colder air
near ground:     warmer air
```

During an inversion:

```text
higher altitude: warmer air
near ground:     colder air
```

The warmer layer above acts like a lid. It suppresses vertical mixing, allowing fog, moisture, smoke, and pollutants to remain trapped close to the surface.

In Nepal valleys:

- the ground cools rapidly at night
- cold dense air drains down slopes
- cold air pools on the valley floor
- warmer air remains above
- pollutants and moisture are trapped below the inversion

Typical inversion heights in this context:

- Shallow radiation inversion: roughly `10-100 m` above ground
- Stronger winter valley inversion: roughly `100-500 m` above the valley floor
- Deep valley or basin inversion: sometimes `500-1000 m`

For this project, "higher altitude" in the inversion discussion usually means **tens to hundreds of meters above the ground or valley floor**, not necessarily high mountains or the upper atmosphere.

## ERA5-Land Relevance

ERA5-Land is relevant for surface fog-favorable conditions, but it is not sufficient by itself for full inversion or airport-visibility modeling.

Useful ERA5-Land variables include:

- `2m_temperature`
- `2m_dewpoint_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `surface_pressure`, if available
- `skin_temperature`
- `surface_sensible_heat_flux`
- `surface_latent_heat_flux`
- `surface_net_thermal_radiation`
- `surface_solar_radiation_downwards`
- `total_precipitation`
- `volumetric_soil_water_layer_1`

ERA5-Land is useful for:

- near-surface saturation
- dewpoint spread
- wind speed
- surface cooling
- surface wetness
- daily or hourly cooling tendencies

ERA5-Land is not enough for:

- true inversion depth
- vertical temperature profile
- fog top height
- boundary-layer vertical structure
- airport-scale runway visibility

For later inversion-focused work, combine ERA5-Land with ERA5 pressure levels or model levels, especially temperature, humidity, and wind profiles, plus boundary-layer-height and cloud-cover variables from ERA5 single levels.

## Key ERA5-Land Fog Variables

The five highest-priority variables from the conversation are:

```text
dewpoint_spread_2m
rh_2m
wind_speed_10m
theta_v_2m
theta_v_delta_1d
```

These are not magic fog variables. They are compact ERA5-Land-only proxies for the main physical ingredients of fog:

```text
moisture near saturation
surface cooling
weak wind / weak turbulence
stable boundary layer / inversion tendency
```

## Variable Explanations

### `dewpoint_spread_2m`

Definition:

```text
dewpoint_spread_2m = T2m - Td2m
```

where:

- `T2m` is 2 m air temperature
- `Td2m` is 2 m dewpoint temperature

Meaning:

- Large spread means the air is relatively dry.
- Small spread means the air is close to saturation.
- A spread near `0 deg C` means condensation or fog is possible if other conditions support it.

Importance:

This is usually one of the most direct fog indicators because fog forms when near-surface air approaches saturation.

### `rh_2m`

Definition:

```text
rh_2m = 100 * es(Td2m) / es(T2m)
```

where `es()` is saturation vapor pressure.

Meaning:

- Low RH means air is too dry for fog.
- RH near `95-100%` indicates fog-favorable saturation.
- Fog or mist may sometimes occur at slightly lower reported RH because of measurement height, aerosols, or grid-scale averaging.

Importance:

Near-surface relative humidity directly represents how close the air is to saturation and is widely used in fog diagnosis and visibility classification.

### `wind_speed_10m`

Definition:

```text
wind_speed_10m = sqrt(u10^2 + v10^2)
```

where:

- `u10` is the 10 m zonal wind component
- `v10` is the 10 m meridional wind component

Meaning:

- Calm or light wind favors radiation fog onset and cold-air pooling.
- Very strong wind usually mixes the boundary layer and prevents shallow fog.
- Weak to moderate mixing can sometimes help fog deepen after onset.

Importance:

Wind speed controls turbulence, mixing, advection, cold-air pooling, and fog dissipation.

### `theta_v_2m`

Definition:

```text
theta = T * (1000 / p)^0.286
theta_v = theta * (1 + 0.61q)
```

where:

- `T` is air temperature in Kelvin
- `p` is pressure in hPa
- `q` is specific humidity
- `theta` is potential temperature
- `theta_v` is virtual potential temperature

Meaning:

`theta_v` is a buoyancy and stability variable. In vertical profiles, an increase of virtual potential temperature with height indicates stable stratification.

Importance:

ERA5-Land only gives near-surface fields, so `theta_v_2m` alone cannot prove an inversion. However, it is still useful as a near-surface moist-stability proxy, especially when combined with cooling tendency, wind speed, and humidity.

### `theta_v_delta_1d`

Definition:

```text
theta_v_delta_1d = theta_v_2m_today - theta_v_2m_previous_day
```

Meaning:

- Negative values suggest near-surface cooling relative to the previous day.
- Positive values suggest warming or reduced cooling.

Importance:

This is a daily proxy for pre-fog cooling and stabilization. It is weaker than the 2-3 hour tendency supported by fog-onset studies, but it is practical for the current daily dataset.

For a future hourly fog model, replace or supplement it with:

```text
theta_v_delta_2h
theta_v_delta_3h
```

## Literature Support

### Dewpoint Spread And Relative Humidity

Fog formation requires near-surface air to approach saturation. Dewpoint depression, or temperature-dewpoint spread, indicates how close the air is to saturation. A rule-based radiation fog study explicitly discusses dewpoint depression and wind speed as paired radiation-fog rules.

Reference:

- Weston et al., "A rule-based method for diagnosing radiation fog in an arid region from NWP forecasts"  
  https://www.sciencedirect.com/science/article/abs/pii/S0022169421002365

### Relative Humidity, Wind, Radiation, And Temperature Trend

A Menut-style observational fog-prediction method identifies four key pre-fog diagnosis variables:

```text
relative humidity
net radiation
10 m wind speed
3 h air-temperature trend
```

This supports using `rh_2m`, `wind_speed_10m`, and a temperature or stability tendency variable in a fog-oriented feature set.

Reference:

- "Understanding and Reducing False Alarms in Observational Fog Prediction"  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6208920/

### Virtual Potential Temperature And Stable Pre-Fog Conditions

The Budapest radiation-fog case study analyzed vertical profiles of virtual potential temperature, wind speed, and relative humidity before, during, and after fog. It found stable atmospheric conditions before fog onset and identified inversion structure using virtual potential temperature profiles.

Reference:

- "An Observational Case Study of a Radiation Fog Event"  
  https://link.springer.com/article/10.1007/s00024-024-03498-w

### Fog Forecasting As A Boundary-Layer Problem

Fog decision-support and review papers emphasize that fog forecasting depends on stability, radiation balance, moisture availability, turbulence, wind/advection, land-surface effects, topography, and boundary-layer structure.

References:

- "Fog Decision Support Systems: A Review of the Current Perspectives"  
  https://www.mdpi.com/2073-4433/14/8/1314
- "Observation, Simulation and Predictability of Fog: Review and Perspectives"  
  https://www.mdpi.com/2073-4433/12/2/235
- "Improvement of numerical weather prediction model analysis during fog conditions through the assimilation of ground-based microwave radiometer observations: a 1D-Var study"  
  https://amt.copernicus.org/articles/13/6593/2020/amt-13-6593-2020.html

## Practical Ranking For This Project

For the current daily offset-prediction dataset:

```text
1. dewpoint_spread_2m
2. rh_2m
3. wind_speed_10m
4. theta_v_2m
5. theta_v_delta_1d
```

For a future hourly fog-onset model:

```text
1. dewpoint_spread_2m
2. rh_2m
3. wind_speed_10m
4. theta_v_delta_2h / theta_v_delta_3h
5. theta_v_2m
```

## Limitation

The current daily setup can only approximate the 2-3 hour pre-fog signal discussed in the literature. A true fog-formation model should eventually move to hourly ERA5-Land and observed fog/visibility labels.
