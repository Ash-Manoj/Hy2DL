# import necessary packages
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from hy2dl.datasetzoo.basedataset import BaseDataset

class UnifiedCAMELSDE_CARAVAN(BaseDataset):
    """Unified dataset class for CAMELS-DE and CARAVAN.
    
    This class integrates catchment data from both CAMELS-DE and CARAVAN datasets.
    
    It maps CAMELS-DE gauge IDs to their corresponding CARAVAN gauge IDs (prefixed with 'camelsde_')
    and merges static attributes and time-series data from both datasets.
    """
    
    def __init__(
        self,
        dynamic_input: Union[List[str], Dict[str, List[str]]],
        target: List[str],
        sequence_length: int,
        time_period: List[str],
        path_camelsde: str,
        path_caravan: str,
        path_entities: str = None,
        entity: str = None,
        check_NaN: bool = True,
        path_additional_features: Optional[str] = "",
        predict_last_n: Optional[int] = 1,
        static_input: Optional[List[str]] = None,
        conceptual_input: Optional[List[str]] = None,
        custom_freq_processing: Optional[Dict[str, int]] = None,
        dynamic_embedding: Optional[bool] = False,
        unique_prediction_blocks: Optional[bool] = False,
    ):
        self.path_camelsde = Path(path_camelsde)
        self.path_caravan = Path(path_caravan)
        
        super(UnifiedCAMELSDE_CARAVAN,self).__init__(
            dynamic_input=dynamic_input,
            target=target,
            sequence_length=sequence_length,
            time_period=time_period,
            path_data=path_camelsde,  # Default to CAMELS-DE path
            path_entities=path_entities,
            entity=entity,
            check_NaN=check_NaN,
            path_additional_features=path_additional_features,
            predict_last_n=predict_last_n,
            static_input=static_input,
            conceptual_input=conceptual_input,
            custom_freq_processing=custom_freq_processing,
            dynamic_embedding=dynamic_embedding,
            unique_prediction_blocks=unique_prediction_blocks,
        )
    
    def _map_gauge_id(self, gauge_id: str) -> str:
        """Maps a CAMELS-DE gauge ID to its corresponding CARAVAN gauge ID."""
        return f"camelsde_{gauge_id}"  # CARAVAN uses this naming convention
    
    def _read_attributes(self) -> pd.DataFrame:
        """Reads and merges catchment attributes from CAMELS-DE and CARAVAN."""
        # Read CAMELS-DE attributes
        camelsde_attrs = []
        read_files = list(self.path_camelsde.glob("*_attributes.csv"))
        
        for file in read_files:
            df = pd.read_csv(file, sep=",", header=0, dtype={"gauge_id": str})
            df.set_index("gauge_id", inplace=True)
            camelsde_attrs.append(df)      

        camelsde_attrs = pd.concat(camelsde_attrs, axis=1)      

        # Encode categorical attributes in case there are any
        for column in camelsde_attrs.columns:
            if camelsde_attrs[column].dtype not in ["float64", "int64"]:
                camelsde_attrs[column], _ = pd.factorize(camelsde_attrs[column], sort=True)

        # Read CARAVAN attributes
        dfs = []
        subdataset_dirs = [d for d in (self.path_caravan / "attributes").glob("*") if d.is_dir()]

        for subdataset_dir in subdataset_dirs:  # Loop over each sub directory
            dfr_list = []
            for csv_file in subdataset_dir.glob("*.csv"):  # Loop over each csv file
                dfr_list.append(pd.read_csv(csv_file, index_col="gauge_id"))
            dfr = pd.concat(dfr_list, axis=1)
            dfs.append(dfr)

        # Merge all DataFrames along the basin index.
        caravan_attrs = pd.concat(dfs, axis=0)

        # Encode categorical attributes in case there are any
        for column in caravan_attrs.columns:
            if caravan_attrs[column].dtype not in ["float64", "int64"]:
                caravan_attrs[column], _ = pd.factorize(caravan_attrs[column], sort=True)
        
        # Rename CARAVAN indices to match CAMELS-DE
        caravan_attrs.index = caravan_attrs.index.str.replace("camelsde_", "")
        
        # Merge datasets on gauge_id
        merged_attrs = camelsde_attrs.join(caravan_attrs, how="outer", rsuffix="_caravan")
        
        return merged_attrs.loc[self.entities_ids, self.static_input]
    
    def _read_data(self, catch_id: str) -> pd.DataFrame:
        """Reads and merges time-series data from CAMELS-DE and CARAVAN for a given catchment."""
        caravan_id = self._map_gauge_id(catch_id)  # Get CARAVAN equivalent ID
        
        # Read CAMELS-DE time-series data
        camelsde_ts = pd.read_csv(self.path_camelsde / "timeseries" / f"CAMELS_DE_hydromet_timeseries_{catch_id}.csv", parse_dates=["date"], index_col="date")
        
        # Read CARAVAN time-series data
        subdataset_name = caravan_id.split("_")[0].lower()
        caravan_filepath = self.path_caravan / "timeseries" / "csv" / subdataset_name / f"{caravan_id}.csv"
        
        if caravan_filepath.exists():
            caravan_ts = pd.read_csv(caravan_filepath, parse_dates=["date"], index_col="date")
            # Merge time-series data
            merged_ts = camelsde_ts.join(caravan_ts, how="outer", rsuffix="_caravan")
        else:
            merged_ts = camelsde_ts  # Use only CAMELS-DE data if CARAVAN data is missing
        
        return merged_ts