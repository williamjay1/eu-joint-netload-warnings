# Data and third-party licences

Author-produced research outputs in this repository, including derived forecast probabilities, joint event labels, forecast-driven review plans, process identifiers, sufficient statistics, aggregate results and figure/table source data, are made available under Creative Commons Attribution 4.0 International (CC BY 4.0). Copyright 2026 Junjie Zhang, to the extent copyright or database rights apply. Cite the repository and associated study and indicate changes: https://creativecommons.org/licenses/by/4.0/.

The MIT licence for author-written software does not replace source-provider data licences or third-party notices. Author-generated outputs are processed research products and are not official products of the source providers. No provider endorses this study.

Source attributions:

- Germany–Luxembourg: Bundesnetzagentur | SMARD.de; market data under CC BY 4.0. Terms: https://www.smard.de/home/datennutzung.
- France: RTE, eco2mix-national-cons-def; Licence Ouverte v2.0 (Etalab). Source metadata: https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def. The source snapshot was collected on 29 September 2026; its metadata records modified 30 July 2026 and data_processed 28 September 2026 22:06:42 UTC. These clocks describe the preserved source version, not authenticated historical first publication. Licence: https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf.
- Belgium: Elia, ODS001/ODS031/ODS032; CC BY 4.0 under the Elia Open Data Licence: https://opendata.elia.be/pages/licence/. Early PV observations use a separately documented, cross-checked third-party archive of the Elia ODS032 export, rather than a current official historical API response: https://raw.githubusercontent.com/jamesdeluk/random-projects/main/solar-pv-time-series-forecasting/ods032.zip.
- Denmark: Source: Energinet (www.energidataservice.dk), ProductionConsumptionSettlement and aggregate price associations based on Elspotprices / DayAheadPrices. Portal terms are CC BY 4.0: https://www.energidataservice.dk/terms-and-conditions. Source datasets and fields are identified in the provenance record.
- Weather: NOAA Global Ensemble Forecast System (GEFS), distributed through NOAA Open Data Dissemination. Terms and attribution: https://registry.opendata.aws/noaa-gefs/ and https://www.weather.gov/disclaimer/. NOAA source material is not assigned new ownership or represented as CC BY licensed by this author. Weather extraction/interpolation and all fitted forecasts are processed research products.
- Spatial masks: Natural Earth Admin 0 – Countries, version 5.1.1, public-domain source geometry: https://www.naturalearthdata.com/about/terms-of-use/.

Transformations include temporal alignment and aggregation, source-specific accounting, net-load construction, seasonal event thresholds, weather extraction/interpolation, model fitting/calibration, event labelling, review-plan construction and statistical summarisation. Consult the methods, source map and schema for exact definitions; transformations differ by source and are not applied uniformly.

This release does not contain raw provider archives, original MW/weather-grid/price time series, model objects or credentials. Source records remain governed by their original terms. Data are revised-archive delayed replay data; publication of these outputs does not certify original real-time availability.

Third-party plotting styles: SciencePlots, commit b9b16959570bd2fbc9ff5118bacc423c3bddd592, MIT, Copyright (c) 2018 John Garrett. Preserve the accompanying licence text and attribution: https://github.com/garrettj403/SciencePlots.

Licence checks were completed on 2 October 2026. Provider metadata and current terms can change; the source snapshot identity and processing provenance are retained in this release.
