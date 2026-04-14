# 🗺️ Gate Shapefiles

Questa cartella contiene i file shapefile per definire i "gate" degli stretti artici usati nell'analisi altimetrica.

## File Disponibili

### Stretti Principali
| File | Descrizione | Satellite/Pass |
|------|-------------|----------------|
| `fram_strait_S3_pass_481.shp` | Stretto di Fram | Sentinel-3, Pass 481 |
| `davis_strait.shp` | Stretto di Davis | Generic |
| `bering_strait_TPJ_pass_076.shp` | Stretto di Bering | TOPEX/Jason, Pass 076 |
| `denmark_strait_TPJ_pass_246.shp` | Stretto di Danimarca | TOPEX/Jason, Pass 246 |
| `nares_strait.shp` | Stretto di Nares | Generic |

### Passaggi Canadesi
| File | Descrizione |
|------|-------------|
| `lancaster_sound.shp` | Lancaster Sound |
| `jones_sound.shp` | Jones Sound |

### Confini Mare-Mare
| File | Regioni |
|------|---------|
| `barents_sea-central_arctic_ocean.shp` | Barents ↔ Artico Centrale |
| `barents_sea-kara_sea.shp` | Barents ↔ Kara |
| `barents_sea_opening_S3_pass_481.shp` | Apertura Barents (S3) |
| `beaufort_sea-canadian_arctic_archipelago.shp` | Beaufort ↔ Arcipelago Canadese |
| `beaufort_sea-central_arctic_ocean.shp` | Beaufort ↔ Artico Centrale |
| `canadian_arctic_archipelago-central_arctic_ocean.shp` | Arcipelago ↔ Artico Centrale |
| `east_siberian_sea-beaufort_sea.shp` | Est Siberiano ↔ Beaufort |
| `east_siberian_sea-central_arctic_ocean.shp` | Est Siberiano ↔ Artico |
| `kara_sea-central_arctic_ocean.shp` | Kara ↔ Artico Centrale |
| `kara_sea-laptev_sea.shp` | Kara ↔ Laptev |
| `laptev_sea-central_arctic_ocean.shp` | Laptev ↔ Artico Centrale |
| `laptev_sea-east_siberian_seas.shp` | Laptev ↔ Est Siberiano |
| `norwegian_sea_boundary_TPJ_pass_220.shp` | Confine Mare Norvegese |

## Formato

Tutti i file sono in formato **ESRI Shapefile** (.shp).

Per caricarli usa:
```python
import geopandas as gpd

gate = gpd.read_file("gates/fram_strait_S3_pass_481.shp")
print(gate.geometry)
```

## Utilizzo

I gate vengono usati per:
1. Filtrare i dati altimetrici lungo specifici transetti
2. Calcolare il trasporto di volume attraverso gli stretti
3. Analisi della pendenza DOT (Dynamic Ocean Topography)
