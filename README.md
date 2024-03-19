# ou3-utils

## Leitura da Base de Dados facilitada

Código: 

import pandas as pd

db_link = "https://github.com/Uchoa-Gui/ou3-utils/raw/main/databank_properties.pickle"

dados, all_units = pd.read_pickle(db_link)
