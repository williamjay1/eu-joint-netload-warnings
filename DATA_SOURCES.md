# Source access and measurement boundaries

The study uses provider-native revised electricity observations and archived NOAA GEFS forecasts. Original historical release vintages were not recovered; delayed replay is an evaluation convention. Endpoints and versions may change after the original snapshots. Hashes identify the exact local inputs in the study manifests; fresh acquisition can yield different revised values.

- SMARD Germany–Luxembourg download documentation: https://www.smard.de/en/downloadcenter/download-market-data — acquisition in `scripts/data_download_observations.py`; native treatment in `data_build_native_v3.py`.
- RTE éCO2mix regional/national open data: https://opendata.reseaux-energies.fr/ — the exact anonymous dataset endpoints are recorded in `data_download_observations.py`; whole-country continental scope and timestamp conventions must be retained.
- Elia open data: https://opendata.elia.be/ — relevant load and generation endpoints in `data_download_observations.py`; this control area includes Sotel and is not a perfectly harmonised three-country accounting partition.
- Energinet settlement schema: https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement and terms https://www.energidataservice.dk/terms-and-conditions — acquisition/accounting/region IDs in `upgrade_data_audit_20261002.py`. Gross-minus-uncertain-self-consumption accounting must be retained; current metadata describes 9–15-day first publication and revisions up to two years.
- NOAA GEFS archive: https://registry.opendata.aws/noaa-gefs/ — immutable range-download source in `gefs_download.py`; common integer-degree extraction, native resolution change and hour interpolation in `gefs_regional_features.py` and `upgrade_data_audit_20261002.py`.
- Natural Earth mask source: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/ — approximate local Danish mask construction is implemented in the data audit code. It is not a power-capacity or exclusive-economic-zone mask.

Check current provider terms before redistributing any source or derived material. This file does not assert that all source licences are identical or that publication of all derivatives is automatically authorised.
