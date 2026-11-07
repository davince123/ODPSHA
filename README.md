# Orientation-Dependent Probabilistic Seismic Hazard Analysis Using Empirical Green's Functions
The code for Orientation-Dependent Probabilistic Seismic Hazard Analysis based on Stochastic Summation of Empirical Green’s Function Simulations
The seven scripts construct earthquake scenarios from potential seismic source zones and magnitude bins, use empirical Green's functions (EGFs) to simulate paired horizontal ground motions, and calculate annual exceedance rates for fixed geographic azimuths and RotD measures. Step 7 extracts return-period intensities and disaggregates hazard by scenario and direction.

## Directory and execution order

```text
rotd-psha/
  step01_scenarios.py
  step02_stf.py
  step03_egf.py
  step04_simulation.py
  step05_intensity.py
  step06_hazard.py
  step07_postprocess.py
  README.md
```

Keep all seven scripts in the same directory. The workflow was verified with Python 3.13. Run this command in Anaconda to install the dependencies.

```powershell
python -m pip install "h5py>=3.10,<4" "numba>=0.61,<0.66" "numpy>=1.26,<2.5" "openpyxl>=3.1,<4" "pandas>=2.1,<4" "pyproj>=3.6,<4" "scipy>=1.11,<2" "shapely>=2.0,<3"
```



## 1. Construct earthquake scenarios

| Configuration key | Input | Required columns or structure |
| --- | --- | --- |
| `events_csv` | Small-earthquake EGF event table | `event_id,zone_id,lon,lat,egf_mw,egf_fc_hz`. Provide each event's source zone, moment magnitude, and corner frequency. |
| `zones_geojson` | Potential seismic source zone boundaries | A WGS84 `FeatureCollection` with `properties.zone_id` on each `Polygon` or `MultiPolygon` feature. |
| `weights_csv` | Existing Voronoi weights | Can replace the boundary file. Accepted columns are `event_id,zone_id,weight` or `Event,ID_pss,Weight_Within_PSS`. The weights must sum to one within each source zone. |
| `recurrence_csv` | Gutenberg–Richter (G–R) parameters | `zone_id,a,b,years,mw_min,mw_max,dm`. Use the fitted coefficients `a` and `b`, with `years` equal to the catalogue duration represented by those coefficients. |
| `rates_csv` | Existing annual rates by magnitude bin | `zone_id,mw,annual_rate`. Use the bin centre for `mw` and rates in yr⁻¹ for `annual_rate`. |
| `rates_xlsx` | Workbook of annual rates by magnitude bin | Each `pss_<zone_id>` worksheet has no header. Its first two columns contain the bin centre and annual occurrence rate. |

Choose one of the three rate inputs. With G–R parameters, the script subtracts the cumulative count at the upper bin edge from the count at the lower edge and divides by `years`. Supply coefficients fitted to a catalogue that has undergone declustering and completeness analysis.

When constructing Voronoi cells, `scenarios.projected_crs` defaults to a local azimuthal equidistant projection centred on the mean coordinates of the EGF events and source zone boundaries. You can specify another projected coordinate reference system in metres. The script clips the cells to their source zone and allocates the magnitude-bin rate in proportion to the area of each clipped cell.

The outputs are `01_scenarios/weights.csv` and `01_scenarios/scenarios.csv`. When the script constructs the cells, `cell_wkt` contains their geometry in projected coordinates.

### Implementation

- `events()` validates event identifiers, coordinates, and small-earthquake source parameters. `magnitude_rates()` reads supplied rates or calculates them from the G–R relation. The final bin ends at `mw_max`.
- `voronoi_weights()` builds and clips cells separately for each source zone, checks the combined cell area, and divides each cell area by the total cell area in that zone.
- `run()` checks that the weights and occurrence rates cover the same zones and pairs each EGF event with the magnitude bins for its zone. `annual_rate` is the rate for the entire zone and magnitude bin. `scenario_rate` is that rate multiplied by the cell weight. Later steps load this table with `read_scenarios()`.
- This file also contains the shared configuration loader `Project`, command-line handler `run_cli()`, table validation, and HDF5 writing functions. Importing these functions from another step loads their definitions. Running the file directly executes step 1. [step01_scenarios.py](step01_scenarios.py)

## 2. Generate stochastic source time functions

This step reads `scenarios.csv` from step 1. Choose one of the following STF input formats.

| Configuration key | Input format |
| --- | --- |
| `fc_samples_csv` | `mw,sample_id,fc_hz` with optional `weight`. Provide at least one candidate corner frequency for each magnitude. Omit `weight` to use equal weights. |
| `legacy_res_point_csv` | A headerless matrix of corner frequencies with one column per magnitude. The default column order is ascending magnitude. Set `stf.res_point_magnitudes` to specify the order explicitly. |
| `legacy_stf_dir` | Existing files at `<event_id>/<mw>/<sample_id>/STF0.txt`. The script reads the target corner frequency and sampling interval from the header, then reads the STF pulse sequence. |

To generate an STF, the script synthesizes pulses using two stages of stochastic delays. By default, it screens candidates against the measured EGF corner frequency and uses `exp(−0.923669 × egf_mw + 4.292291)` as the EGF corner frequency for synthesis. Set `stf.egf_fc_model` to `measured` to use the measured value for both operations. The target magnitude must exceed the EGF magnitude by more than 0.1. The target corner frequency must be below the measured EGF value and at most equal to the EGF value used for synthesis. `stf.seed` sets the random seed, and `stf.delay_repetitions` sets the number of independent delay samples per corner frequency.

The outputs are `02_stf/<event_id>/<mw>.h5` and `02_stf/manifest.csv`. Sample weights are normalized within each scenario.

### Implementation

- `read_fc_samples()` groups candidate corner frequencies by magnitude. `generate_stfs()` screens those candidates for each scenario. It derives a seed from the configured seed, event identifier, magnitude, candidate identifier, and repetition index.
- `synthesize()` calculates the target-to-EGF seismic moment ratio, the pulse count for each stage, and the scaling factor. `delay_grid()` uses numerical integration to tabulate the delay distribution. `_tgen_fast_numba()` draws delays by rejection sampling. `_generate_rs_numba()` combines the two stages of delays and accumulates pulses on the time grid. The script trims leading and trailing zeros and rounds pulse values to six decimal places.
- `import_stfs()` reads existing STF files, checks the sampling interval in each header, and preserves the corner frequency. `run()` writes `sample_id`, `fc_hz`, normalized `weight`, and `stf/<index>`. Scenarios without a usable STF are omitted from the manifest. The coverage table in step 6 identifies the corresponding missing cases. [step02_stf.py](step02_stf.py)

## 3. Transfer empirical Green's functions

| Configuration key | Input file and columns |
| --- | --- |
| `sites_csv` | `site_id,lon,lat` for the target sites where hazard is evaluated. |
| `stations_csv` | `station_id,lon,lat` for the reference stations. |
| `records_csv` | `event_id,station_id,e_file,n_file` with optional `format`, which defaults to `txt`. Paired E and N records must have equal lengths, share a common time reference, and use the configured `dt_s`. |
| `attenuation_txt` | Path attenuation table. The first line contains metadata. The second starts with `Dist_[km]` followed by frequency columns. Subsequent rows give distance and base-10 log attenuation at each frequency. |
| `site_spectra_csv` | `location_id,file`. Each identifier refers to a reference station or target site. Spectrum files can be two-column CSVs with `Frequency_Hz,Site_Amplification`, or `.site` files with one metadata line followed by space-separated frequencies and amplification factors. |
| `legacy_egf_dir` | Root directory of already transferred EGFs. Files under `<site_id>/<event_id>/` are paired by the suffixes `_<station_id>_GM_E.txt` and `_<station_id>_GM_N.txt`. This mode needs only `sites_csv`, the scenario table, and this directory. |

Use a single column without a header for `txt` waveforms. For frequency-domain records, set `format=fas` and supply `frequency_Hz,amplitude,phase_rad` in full FFT order. The script uses colocated records when available. Otherwise, it sorts stations by distance to the target and selects the five nearest available reference stations by default. For transfer, supply amplification spectra for the reference stations and target sites.

The script applies the same complex transfer function to both components of each reference record. Its amplitude is the product of the Brune source spectrum ratio, the path attenuation ratio, and the site amplification ratio. Set `egf.band_method` to `linear` to interpolate gains within the frequency band and use unity gain outside it. Set `egf.travel_time` to `true` to apply a phase correction for the travel-time difference.

| `egf` key | Default | Meaning |
| --- | --- | --- |
| `nearest_donors` | `5` | Maximum number of reference stations per site and event. |
| `max_event_distance_km` | `500` | Maximum distance from the source to the target site. |
| `direct_tolerance_km` | `0.001` | Distance tolerance for matching a station to a colocated target site. |
| `velocity_ratio` | `0.8` | Ratio of small-earthquake rupture speed to shear-wave speed. |
| `small_event_rupture_azimuth_deg` | `0` | Small-earthquake rupture azimuth, measured clockwise from north. |
| `band_method` | `legacy_bins` | Takes the gain at the first grid frequency at or above the query frequency. Above the highest frequency, the gain is one. The other option is `linear`. |
| `travel_time` | `false` | Enables the travel-time phase correction. |
| `shear_velocity_km_s` | `3.5` | Shear-wave speed used to calculate the travel-time difference. |

The outputs are `03_egf/<site_id>/<event_id>.h5` and `03_egf/manifest.csv`.

### Implementation

- `waveform()` reads acceleration from a TXT file or reconstructs a real waveform from Fourier amplitudes and phases. `read_pair()` checks that the E and N components have equal lengths. `legacy_pairs()` matches already transferred components by station identifier and component suffix.
- `Transfer.apply()` calculates distances and azimuths from the source to the reference station and target site. `apparent_fc()` adjusts the corner frequencies for azimuth, and `brune_ratio()` calculates their source spectrum ratio. The path correction is 10 raised to the difference between the target and reference log attenuation values. The site correction is the ratio of the target to reference site amplification.
- `fourier_transfer()` applies the same amplitude ratio and phase factor to both Fourier spectra, then transforms them back to the time domain. With the travel-time correction enabled, the phase factor is `exp(−2π i f Δt)`, where `Δt = (target distance − reference distance) / shear-wave speed`. Zero padding gives the time shift room at the record boundaries.
- `run()` processes each site and event and selects colocated records or reference stations. `donor_id` identifies each selected station. `egf/<index>` stores two rows in E, N order. [step03_egf.py](step03_egf.py)

## 4. Simulate paired horizontal ground motions

This step reads the manifests and HDF5 archives from steps 2 and 3 and joins them by event identifier. For each site, event, and target magnitude, the same STF is convolved with both the E and N components from a reference station. When `simulation.target_directivity` is `true`, the script applies a Brune spectrum ratio after convolution using the target corner frequency and the source-to-site azimuth. Set `simulation.rupture_azimuth_deg` to change the default rupture azimuth of 60°.

Target directivity is enabled by default. `simulation.velocity_ratio` defaults to 0.8, and `simulation.band_method` defaults to `legacy_bins`. These settings apply to the target earthquake and are configured separately from the small-earthquake transfer in step 3.

The outputs are `04_motion/<site_id>/<mw>/<event_id>/motion.h5` and `04_motion/manifest.csv`. Each STF's weight is divided equally among the reference stations used for that scenario. Ground motions are stored as floating-point values by default. Set `simulation.int16` to `true` to store compressed integers with scale factors.

### Implementation

- `run()` joins the two manifests, checks that their `dt_s` values agree, and iterates over STF samples and reference stations at each site and magnitude.
- `simulate()` computes the full convolution for both components. The output length is `EGF length + STF length − 1`. When target directivity is enabled, the function applies the Brune ratio using the Fourier transfer routines from step 3. Step 3 transfers a reference small-earthquake recording. Step 4 adjusts the spectrum of the simulated target earthquake at the target azimuth.
- `write_motion()` stores each record in `acceleration/<index>`. `record_names` identifies STF and reference station combinations, and `weights` stores their sample weights. In integer mode, multiply each component by its corresponding `scale` to recover the acceleration values. [step04_simulation.py](step04_simulation.py)

## 5. Calculate rotational intensity measures

This step reads the step 4 manifest and motion archives. It computes PGA and Sa at 1° intervals from 0° through 179° using the E and N components. The script uses Newmark integration to calculate the **absolute acceleration response spectrum**. The defaults are 5% damping and periods of 0.5 s and 2.0 s. Set `intensity.periods_s` and `intensity.damping` to change these values.

The raw rotation angle `α` follows `a(α) = E cos α + N sin α`. The corresponding geographic axis azimuth, measured clockwise from north, is `(90 − α) mod 180`. For example, geographic azimuth 145° corresponds to raw angle 125° in the HDF5 archive.

The outputs are `05_im/<site_id>/<mw>/<event_id>/RotD_IM.h5` and `05_im/manifest.csv`. The `PGA` dataset has dimensions angle × sample, and `Sa` has dimensions angle × period × sample. The archive also stores `angles_deg`, `periods`, `record_names`, and `weights`.

### Implementation

- `periods()` checks that the periods are positive and strictly increasing. `newmark_constants()` computes the integration coefficients from the period, damping ratio, and time step using β = 0.25 and γ = 0.5.
- `rotd_pga_and_sa_numba()` forms the acceleration time series at each angle and takes its peak absolute value as PGA. For each period, it starts from zero relative displacement and velocity, advances the relative motion in time, and takes the maximum absolute value of relative acceleration plus ground acceleration as Sa.
- `run()` reads each paired record from step 4 and preserves its sample weight. `write_im()` stores the IM arrays. When `intensity.int16` is enabled, it stores a `scale` attribute for each array. Step 6 uses the raw angle index corresponding to the requested geographic azimuth. [step05_intensity.py](step05_intensity.py)

## 6. Integrate hazard curves

This step reads the scenario table from step 1, the target site table, and the IM archives from step 5. Alternatively, set `legacy_im_dir` to read existing archives at `<site_id>/<mw>/<event_id>/RotD_IM.h5`. Integer PGA and Sa are restored by multiplying by the dataset's `scale` attribute. Archives without sample weights are treated as equally weighted.

Set `hazard.im_grid` to `[minimum, maximum, number of levels]` and `hazard.fixed_azimuths_deg` to the desired integer geographic azimuths. RotD00, RotD50, and RotD100 are the minimum, median, and maximum across rotation angles for each sample. The empirical method adds the weights of samples strictly above the threshold within each scenario, multiplies by the scenario's annual occurrence rate, and sums across scenarios. The lognormal method estimates the mean and standard deviation of natural-log IM values within each scenario. It uses `ddof=1` for equally weighted samples. When there are too few samples, any zero IM values, or zero variance, the script uses empirical exceedance for that scenario and intensity series and records the fallback count.

The default grid is `[0.0001, 10.0, 2000]`, and the default fixed azimuths span 0° to 179°. `hazard.method` accepts `empirical`, `lognormal`, or `both`. The default is `both`.

The curves are written to `06_hazard/<site_id>/<IM>_empirical.csv` and/or `<IM>_lognormal.csv`. The first column, `im_level`, gives the intensity threshold. The remaining columns give annual exceedance rates in yr⁻¹. By default, a missing scenario with a positive occurrence rate raises an error. Set `hazard.allow_missing` to `true` to integrate the available archives using the scenario weights from step 1. `coverage.csv` identifies missing scenarios, and `coverage_summary.csv` reports the total and included occurrence rates.

### Implementation

- `cases()` associates sites and scenarios with their IM archives. `read_im()` restores scaled integer values, checks rotation angles and array shapes, selects Sa by period, and normalizes sample weights within each scenario.
- `levels_and_angles()` creates logarithmically spaced intensity thresholds. Fixed-azimuth columns use names such as `fixed_145`. The script computes each sample's RotD statistics before evaluating exceedance.
- `empirical_exceedance()` sorts the samples by IM and sums weights above each threshold. `lognormal_exceedance()` uses the weighted mean and standard deviation of log IM to calculate the probability of exceeding the same threshold.
- `run()` sums the product of scenario occurrence rate and conditional exceedance probability for each site, IM, orientation, and threshold. In the coverage summary, `included_rate / total_rate` gives the fraction of the scenario occurrence rate represented at each site. [step06_hazard.py](step06_hazard.py)

## 7. Extract return levels and disaggregate hazard by direction

This optional step reads the empirical hazard curves from step 6 along with the same site table, scenario table, and IM archives. Set `postprocess.return_periods_years` to the desired return periods. The script selects the intensity grid point whose annual exceedance rate is closest to `1/T` and flags any target rate outside the curve's range. For RotD100, `postprocess.window_indices` sets the number of grid points on either side of the target intensity. The script weights samples in that window by annual occurrence rate, calculates each scenario's contribution, and fits an axial von Mises distribution to their directions. Set `postprocess.angle_convention` to `legacy_pdf` to use the folded-angle convention `−abs(90 − α) mod 180`.

The default return periods are `[50, 475, 2500]` years, the default window half-width is 50 grid points, and the default angle convention is `geographic`.

The outputs in `07_postprocess/` are `return_levels.csv`, which includes site coordinates for mapping, `disaggregation.csv`, `direction_summary.csv`, and `direction_pdf.csv`.

### Implementation

- `read_targets()` extracts the grid intensity for each return period from each empirical hazard curve. It records `outside_grid` when the target exceedance rate lies outside the range covered by the curve.
- `run()` selects RotD100 samples strictly between the window boundaries. Each sample contributes its scenario's annual occurrence rate multiplied by its sample weight. A scenario's `fraction` is its contribution divided by the total contribution within the window. A sample's direction is the first rotation angle at which its IM reaches its maximum.
- `axial_fit()` computes the weighted circular mean of the doubled angles and halves the resulting angle to obtain the mean axis `mu_deg`. It uses the mean resultant length to estimate the concentration `kappa`. `axial_density()` returns a probability density per degree and uses finer angle spacing around narrow peaks. The summary records `empty_window` when the window contains no samples. [step07_postprocess.py](step07_postprocess.py)
