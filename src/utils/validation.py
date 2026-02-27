import numpy as np
from sklearn.cluster import KMeans

def get_spatial_blocks(stations_df, n_blocks=5, random_state=42):
    """
    Assign each station to a spatial block using K-Means clustering on (Lat, Lon).
    Args:
        stations_df: DataFrame with 'StationID', 'Lat', 'Lon'.
        n_blocks: Number of folds/blocks.
    Returns:
        dict: {StationID: block_id}
    """
    coords = stations_df[['Lat', 'Lon']].values
    kmeans = KMeans(n_clusters=n_blocks, random_state=random_state, n_init=10)
    block_ids = kmeans.fit_predict(coords)
    
    station_to_block = {sid: bid for sid, bid in zip(stations_df['StationID'], block_ids)}
    return station_to_block

def get_stations_in_fold(station_to_block, fold_idx, mode='train'):
    """
    Get list of station IDs for a given fold.
    mode='train': return all stations NOT in fold_idx.
    mode='val': return all stations IN fold_idx.
    """
    if mode == 'train':
        return [sid for sid, bid in station_to_block.items() if bid != fold_idx]
    elif mode == 'val':
        return [sid for sid, bid in station_to_block.items() if bid == fold_idx]
    else:
        raise ValueError("Mode must be 'train' or 'val'")
