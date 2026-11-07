# Start Forecast.exe

Forecast.exe 1.0.0 is a local Python/Streamlit application. No public website, hosting,
paid AI service, or API key is required. Install **Python 3.11 or newer** first.
The project name does not mean it is a compiled Windows `.exe` installer.

## Easiest launch

- **macOS:** Open `Launch.command`. If macOS restricts downloaded scripts, run
  `python3.11 launch.py` from this folder in Terminal instead.
- **Windows:** Open `Launch.bat` with Python installed.
- **Linux or any platform:** Run `python launch.py` with Python 3.11+.

The launcher creates `.venv`, installs exact dependencies from `requirements.lock`,
and opens the app in your browser at `http://127.0.0.1:8501`. The first launch needs
internet to install packages; later launches reuse the environment. Keep the terminal
window open. Press Ctrl+C there to stop the app. The launcher binds only to your computer.
`python launch.py --setup-only` installs without starting the app.

## First forecast

1. Open **Weather** for current external weather estimates and a seven-day Open-Meteo
   forecast immediately, without downloading training history or fitting a model.
2. Choose **Use my location** and respond to the browser permission prompt, or open
   **⚙**, search, choose a result, and click **Use selected city**.
   New York City is the clearly labeled default until you choose a city.
3. For your own predictions, open **Model Studio**, load at least two years of complete
   history (five by default), and click **Train my models**. Training runs in the background.
   You can keep viewing external weather, explore other pages, or cancel between horizons.
4. Open **Weather → Local ML** for the model's outlook and uncertainty. **Weather History**
   explores the reference data, **Compare Cities** compares external forecasts for different places; **Model Studio → Saved Runs** compares experiments, and **How It Works**
   explains the methods and limits. Fitting controls and detailed metrics stay in Model Studio.

The external current-temperature estimate is not an ML prediction or necessarily a station
observation. External rain percentages are the daily maximum of hourly probabilities;
local ML estimates the probability of a daily rain event ≥0.1 mm. They are labeled separately.
Weather refreshes approximately every ten minutes while its view is open; use **Refresh weather**
to request an update sooner. The display includes the provider's valid time and retrieval time.

Enable **Synthetic demo** for deterministic generated historical weather when exploring offline.
Every demo forecast and evaluation remains clearly labeled. Network failures never silently
replace real weather with synthetic data. Demo training also lives in Model Studio.

## Location and remembered cities

The app never requests device location on page load and does not use IP geolocation.
Clicking the **triangular location arrow** invokes the browser's native permission flow. If permission
is denied, times out, is blocked, or no nearby city is available, manual search remains available.
Geolocation requires HTTPS or localhost; the supplied local launcher uses localhost.

Device coordinates are rounded to two decimals in the browser and used transiently to
find a nearby populated place using the installed GeoNames city index. The app uses that
city's center for weather, training, and persistence. City names, state/region, country,
and timezone are resolved without sending device coordinates to a reverse-geocoding API.
This is an approximate nearby-city match, not an exact address; use search to correct it.
On a local install this lookup happens on your computer. If you separately host the app,
the app server receives the rounded coordinates for the lookup.

**Remember this city in this browser** saves the selected city, city-center coordinates,
and timezone in browser localStorage. Location and preference controls are in the header’s triangular arrow and gear menu. The next visit restores it without a new location
permission request. Uncheck it to stop remembering, or open **⚙** and click **Forget saved city**. Forgetting removes the browser preference; downloaded
city history and saved model runs remain available. Blocked browser storage does not prevent
manual selection for the current visit. Every city still requires its own locally trained models.

Completed training runs are immutable and saved automatically. Retraining adds a new
run instead of overwriting the old one. The active experiment must match the selected
city, demo/real mode, full dataset fingerprint, and seed. Historical runs remain inspectable.
A model is not retrained on ordinary interactions or when refreshing weather.

**Refresh recent days** requests missing date ranges plus the latest seven days.
Older cached observations are reused. API failures retain the last valid local master.
To request a new live origin, load through yesterday and explicitly train a new experiment.
If an archive is delayed, choose an earlier historical end date to explore historical
analysis; stale sources still disable live forecasts. No synthetic fallback is automatic.

## Manual installation and development

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e '.[dev]'
streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. The universal lock includes exact runtime
and development versions, with platform/Python markers where needed. CI uses the same
lock. To update it intentionally, regenerate with `uv pip compile pyproject.toml --extra dev
--universal --python-version 3.11 --output-file requirements.lock` and rerun checks.

```sh
python scripts/fetch_data.py --city "New York" --years 5
python scripts/fetch_data.py --city "New York" --years 5 --refresh
python scripts/train_model.py --city "New York" --horizons 1 2 3 4 5 6 7 --seed 42
python scripts/train_model.py --demo
python scripts/backtest.py --cities "New York" "London" "Singapore"
python scripts/backtest.py --demo
ruff check .
pytest -q
```

CLI search uses the first geocoding result and prints its identity; use city/country
qualifiers. UI search allows choosing a match. CLI training writes registered runs and
forecast CSVs into `artifacts` by default (`--output` accepts a directory). Run commands
from this folder so CLI and UI share data and experiments. Backtests save metrics,
seasonal diagnostics and provenance under `artifacts/backtests` by default.

## How evaluation works

Twenty-one histogram gradient boosting models predict high/low temperature and rain
occurrence (≥0.1 mm) directly for horizons 1–7. Lagged readings and trailing averages end
at each origin; target-date seasonality is known in advance. Crossing high/low point
estimates are sorted consistently before calibration, scoring and inference.

Default origin partitions are chronological: 65% training, 17% calibration, 18% test,
with seven origins purged before each next period. Base models fit training data only.
Rain probabilities use regularized sigmoid calibration on calibration-period outcomes
when at least 30 calibration cases, both labels, and varying predictions exist;
otherwise the raw prior/probabilities are used and the reason is recorded. Both raw and
calibrated test Brier scores are shown. A lower Brier score alone does not prove better
calibration; inspect the reliability chart too.

Temperature intervals use the 10th/90th calibration residual quantiles by horizon and
meteorological season (DJF/MAM/JJA/SON). Groups below 40 calibration days fall back to
horizon-wide quantiles. These approximate 80% intervals have no coverage guarantee;
actual test coverage and sample counts are shown by season/horizon. Models stay fitted
to training data; there is no silent final refit on calibration/test.

Persistence uses origin temperatures and trailing 30-day rain frequency. Climatology uses
training-only values within ±15 circular calendar days. Every method scores identical
observations per horizon. Overall primary metrics average the seven horizons equally.
Accuracy remains in °C even when forecast presentation uses °F.

Rolling evaluation refits three expanding folds with 90 test origins and 120 calibration
origins each, with purges. Later folds may learn earlier folds’ weather as time advances;
each fold’s own test remains untouched. Each city is fitted independently. Compare cities
only with the displayed date windows, reference source and sample counts in mind.

Live inference requires history ending **yesterday in the selected timezone**. The current
external Open-Meteo comparison overlays only matching valid dates; it never trains models
or scores historical tests. Archived professional-forecast benchmarking is not implemented.

## Data, artifacts and limits

Weather: [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), CC BY 4.0.
Historical references include reanalysis, not necessarily direct station observations.
Geocoding: GeoNames via Open-Meteo. Raw/normalized Parquet and JSON metadata preserve
coordinates, timezone, dates, units and retrieval details. Invalid values/gaps stop training.
The free API is noncommercial; review [current terms](https://open-meteo.com/en/terms)
before public/commercial use. Requests use bounded timeouts, retries and persistent caching.

Only load trusted locally generated joblib files. Full dataset fingerprints, feature
schemas, calibration contents, model/library version, source, dates, seed and scores
are recorded/validated. The public release is 1.0.0. Its internal model-artifact schema remains 2.0.0, so compatible saved models still load. Artifact schema numbers are separate from public release numbers; legacy schema-1 artifacts require retraining.
Experiments with different fingerprints/test windows are explicitly flagged when compared.

Training jobs run while the app process remains open. They are not a durable service;
closing the terminal ends them. Scheduled retraining, precipitation-amount models and
archived NWP benchmarks are not included. These predictions are experimental and must
not guide safety-critical decisions.

Generated `data`, `artifacts`, `.venv`, Python caches and installation metadata are
created locally and excluded from the source delivery. The README, this guide, code,
styles, launcher, dependency definitions, tests and CI files serve distinct purposes.

## Attribution and visual design

Offline nearby-city lookup uses [reverse_geocoder](https://github.com/thampiman/reverse-geocoder)
(LGPL code) and [GeoNames](https://www.geonames.org/) city data (CC BY attribution).
Country names use pycountry. Offline timezone boundaries use [tzfpy](https://github.com/ringsaturn/tzfpy)
(MIT code; ODbL boundary data derived from timezone-boundary-builder). Manual city search
and operational/historical weather still come from Open-Meteo. These libraries are installed
through the dependency lock; no paid API key is needed. Boundary/nearby-city approximations
may differ near borders; manual city selection is always available.

Brand catchphrase: **Decoding the Skies, One Prediction at a Time.**

The editable vector wordmark is `static/forecast-logo.svg`; the matching icon/favicon is
`static/forecast-mark.svg`. Both were authored as SVG, with an orbit, cloud, and prediction
line. `static/atmosphere.png` is decorative artwork generated with the built-in image-generation
tool; it never represents the actual weather or geography of the selected city. The image
is served locally from the app's static folder. The background theme follows the selected city’s current provider weather code and day/night setting: sunny, cloudy, rain, snow, fog, storm, or clear night. Weather refreshes every ten minutes while the Weather view is active. Missing or stale readings and synthetic demo use a neutral theme; the art does not depict actual cloud cover or precipitation intensity. Reduced-motion settings suppress animations.

Final background-generation prompt:

> Create one cinematic atmospheric background asset for a premium dark weather application called Forecast.exe. Use case: stylized-concept. Asset: photorealistic abstract nocturnal cloudscape texture, ultra-wide landscape composition, no buildings or recognizable geography. Deep midnight navy and charcoal, soft three-dimensional blue-gray cloud banks along the bottom and right edges, a subtle pale silver-blue light source behind the clouds at upper right, fine natural film grain, restrained volumetric rays and profound negative space through center and left for highly legible white interface text. Moody large-format cinema photography, tasteful desaturated color grading, realistic texture and depth. No text, lettering, logo, watermark, UI, stars, planets, moon, lightning, or weather symbols. Keep most of the image dark; no bright white areas. This is decorative atmospheric art, not a weather observation. Wide 16:9 composition.

Python tests include offline city resolution, browser-preference validation, permission-denial
fallbacks, and external weather independent of models. Node 22+ is optional for development
and runs the browser-component unit checks in CI; it is not needed to install or run the app.

## Header navigation and city comparison

All navigation is in the header; there is no sidebar. The five centered tabs are **Weather**,
**Weather History**, **Model Studio**, **Compare Cities**, and **How It Works**. Each opens
with a five-sentence introduction. Click the Forecast.exe logo to reload the site and return
to Weather; a remembered city restores normally. The full catchphrase remains visible below
the logo on desktop and mobile.

- **Triangular arrow:** Request device location only when clicked.
- **Temperature pill:** Open the dropdown and choose °C or °F.
- **Gear pill:** Search/change city, manage remembered-city preferences, enable synthetic demo,
  set history dates and seed, and select saved experiments when applicable.

**Compare Cities** compares external current estimates and seven-day forecasts for two to
four cities. Search in the gear menu and click **Add to comparison** to add another place.
The view includes current temperature, humidity, wind, temperature outlooks, rain probabilities,
and a downloadable daily table. Every city keeps its local timezone and valid dates; this is
not a synchronized-UTC or forecast-accuracy benchmark. It never borrows a model trained for
another city. Synthetic demo does not fabricate multi-city weather; disable demo to compare
real external forecasts. Saved model experiments remain under **Model Studio → Saved Runs**.

Chart keys appear in separate panels below comparison graphs so they cannot overlap the date axis. The page footer contains clickable LinkedIn, GitHub, and email icons at the bottom left, with accessible names and no visible URLs.
