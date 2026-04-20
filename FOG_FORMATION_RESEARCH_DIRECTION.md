# Fog Formation: Literature-Based Conclusions and Research Direction

Date: 2026-04-20

## Why this note exists

This note turns the fog papers and discussion notes into a clear conclusion for the project:

- what we can reasonably claim about **fog formation**
- what gap remains for **fog prediction**
- how the current graph-based research can be extended to tackle that gap

## Main conclusion

Fog formation is not controlled by one variable. It is a **multiscale boundary-layer process** driven by the interaction of:

- nocturnal radiative cooling
- weak wind and weak turbulence during onset
- vertical temperature structure and low-level stability
- later changes in turbulence and wind shear
- local land-surface controls such as soil moisture heterogeneity
- mesoscale/topographic controls that determine where fog is likely to occur

For **radiation fog**, the literature consistently suggests:

- fog tends to form at night or late evening under stable or weakly turbulent conditions
- weak turbulence helps the surface cool and saturate
- stronger turbulence later can deepen the fog layer during transition, but sustained turbulence tends to **dissipate** fog
- fog onset, growth, and dissipation are therefore strongly linked to the time evolution of turbulence, inversion strength, moisture availability, and surface energy balance

So the scientific takeaway is:

`Fog prediction should be treated as a spatiotemporal, multiscale, physics-constrained event prediction problem rather than as a simple temperature forecast problem.`

## What the papers support

### 1. Fog life cycle depends on thermodynamics plus turbulence

The SOFOG3D paper shows that fog formation, transition, mature phase, and dissipation are best explained by combining:

- fog liquid water content / liquid water path
- fog depth / fog top height
- wind and turbulence profiles
- temperature profiles
- diagnostics such as fog liquid water reservoir and adiabaticity

It also shows that turbulence is not only a dissipation signal. Moderate mixing can be part of the transition from stable to adiabatic fog, while stronger mechanical and/or thermal turbulence is associated with dissipation.

### 2. Radiation fog onset prefers weak turbulence; dissipation follows turbulence increase

The Budapest observational case study supports a classic radiation-fog picture:

- fog developed upward from the surface
- near-surface turbulence stayed weak during formation
- dissipation occurred when turbulence increased

That means a useful predictor should not only estimate temperature or humidity, but also detect the **timing of turbulence regime change**.

### 3. Mesoscale controls decide where fog forms; microscale land heterogeneity changes duration

The Christchurch simulation paper shows:

- mesoscale meteorology and terrain strongly control the location of fog occurrence
- soil moisture heterogeneity does not strongly change the large-scale fog pattern
- but it can materially change fog duration at small spatial scales

This is important for your work because it implies that a good model must combine:

- large-scale forcing from NWP / reanalysis
- local terrain and land-surface information
- station-scale or grid-scale spatiotemporal interactions

## Defensible research conclusion

From these papers, the strongest conclusion is:

`Fog formation can be predicted better when the model explicitly represents both large-scale meteorological forcing and local boundary-layer processes, especially nighttime cooling, turbulence evolution, stability, moisture availability, and land-surface heterogeneity.`

A second, equally important conclusion is:

`Predicting fog occurrence is not enough; fog onset time, duration, and dissipation are controlled by different processes and should be modeled explicitly.`

## What this means for the current project

Your current repo is a **daily station-level temperature offset correction model**. That is scientifically useful, but it is **not yet a fog-formation model**.

Right now the model predicts:

- `ΔTmax`
- `ΔTmin`

This can help with fog indirectly, because nocturnal cold bias and inversion-sensitive Tmin errors matter for fog risk. But fog itself is usually an **hourly to sub-daily event**, and the literature above says it depends on variables the current pipeline does not represent well enough:

- sub-daily temporal evolution
- humidity / dew point depression
- wind shear and turbulence proxies
- nighttime radiation and surface energy balance
- fog type and vertical structure
- soil moisture heterogeneity

So the correct framing is:

`The current model is a strong precursor for fog-risk research, but not the final fog nowcasting architecture.`

## How your research can tackle fog prediction

### 1. Reframe the target

Move from daily temperature-offset prediction alone to one of these fog targets:

- binary fog occurrence within the next `6-12 h`
- fog onset time
- fog duration
- visibility class or minimum visibility
- multi-task target: `occurrence + onset + duration`

If data are limited, start with:

- `fog / no fog`
- `radiation fog vs cloud-base-lowering fog` if labels exist

### 2. Upgrade from static daily graph to spatiotemporal forecasting

The current graph captures station-to-station structure, but fog requires time history. A better model family would be:

- spatiotemporal graph neural network
- graph Transformer with temporal attention
- a baseline inspired by AirFormer-style spatial + temporal blocks

Practical translation for this repo:

- keep the station graph
- replace single-day inputs with a sequence window, e.g. previous `6-24 h`
- predict fog risk at future lead times, e.g. `+1 h`, `+3 h`, `+6 h`

### 3. Add fog-relevant predictors

The next feature set should prioritize variables that map to the mechanisms identified in the papers:

- 2 m temperature
- dew point or relative humidity
- dew point depression
- wind speed and direction
- wind shear proxies
- pressure tendency if available
- cloud cover / low cloud
- net radiation or shortwave + longwave components
- soil moisture or satellite-derived soil moisture proxy
- topography, slope, valley exposure, cold-air pooling proxy
- seasonality and local time

If turbulence is not directly observed, use proxy variables:

- wind speed changes
- shear between levels
- Richardson-number-like stability proxy if multi-level data exist
- surface sensible heat flux proxy from radiation + temperature tendencies

### 4. Use a two-stage research strategy

The most realistic path for this project is not to jump directly to a full remote-sensing fog nowcaster.

Instead:

### Stage A. Improve meteorological precursors

Extend the current work to correct the near-surface variables that drive fog:

- nighttime temperature
- humidity / dew point
- low-level wind

This gives a strong physics-grounded intermediate result.

### Stage B. Build a fog-event model on top

Use corrected meteorology plus static geography to predict:

- fog occurrence
- onset
- duration

This is easier to defend scientifically than trying to predict fog from raw coarse inputs in one step.

### 5. Treat advanced sensors as ideal inputs, not hard requirements

The SOFOG3D result supports the value of:

- cloud radar
- microwave radiometer
- wind lidar
- surface energy balance
- meteorological stations

But for a sparse-network deployment setting such as Nepal, this instrumentation is usually unavailable.

So a practical research claim should be:

`Best-case fog process understanding comes from vertically profiling instruments, but an operationally useful fog predictor for data-scarce regions should learn from NWP/reanalysis, surface stations, terrain, and satellite-derived land-surface proxies.`

That keeps the project realistic.

## Recommended project framing

The strongest version of the fog direction is:

`A spatiotemporal graph-based model for predicting fog risk from coarse meteorological forcing and local station/terrain information, with improved nighttime temperature-humidity correction as the intermediate physics-aware step.`

In plain terms:

- first solve the local nocturnal bias problem
- then use those corrected local conditions to predict fog

## Suggested concrete research questions

1. Does correcting nighttime temperature, humidity, and wind improve fog-occurrence prediction relative to raw ERA5/NWP?
2. Can a spatiotemporal graph model predict fog onset and dissipation better than station-wise baselines?
3. How much do terrain and soil-moisture heterogeneity improve forecast skill, especially for fog duration?
4. Does the model generalize under sparse-station settings?

## Minimum viable next step for this repo

If you want a tractable next experiment without rebuilding everything:

1. keep the current station graph
2. move from daily to hourly inputs if available
3. add humidity, wind, and nighttime radiation-related predictors
4. create a fog label from visibility or RH-based proxy
5. train a simple spatiotemporal baseline first
6. compare against persistence, raw ERA5 thresholds, and a non-graph sequence baseline

If hourly fog labels are not available yet, the fallback is:

- continue the current Tmin correction work
- position it as **fog-risk precursor modeling**
- show that corrected nocturnal conditions improve a downstream fog proxy

## Bottom line

Your research should not claim that fog is mainly a temperature problem.

The literature supports a stronger and more accurate claim:

`Fog formation is a multiscale interaction between cooling, moisture, stability, turbulence, and land-surface heterogeneity; therefore, fog prediction should combine spatiotemporal meteorology, local terrain/soil controls, and boundary-layer-aware features.`

That gives you a clear path:

- current work: station-level correction of inversion-sensitive meteorology
- next work: spatiotemporal fog-risk prediction
- stronger long-term goal: onset and duration prediction under sparse observations

## Sources

1. Dione et al. (2023), *Role of thermodynamic and turbulence processes on the fog life cycle during SOFOG3D experiment*  
   https://acp.copernicus.org/articles/23/15711/2023/

2. Gandhi et al. (2024), *An Observational Case Study of a Radiation Fog Event*  
   https://link.springer.com/article/10.1007/s00024-024-03498-w

3. Lin et al. (2023), *Investigating multiscale meteorological controls and impact of soil moisture heterogeneity on radiation fog in complex terrain using semi-idealised simulations*  
   https://acp.copernicus.org/articles/23/14451/2023/
