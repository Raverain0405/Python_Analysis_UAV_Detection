from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h5py
from bs4 import BeautifulSoup
import re
import pickle


def get_datasets():
    for i in range(1, 4):
        url = 'http://mason.gmu.edu/~lzhao9/materials/data/UAV/data/pub_dataset{}.mat'.format(
            i)
        r = requests.get(url, allow_redirects=True)
        open('pub_dataset{}.mat'.format(i), 'wb').write(r.content)


def import_dataset():
    full_data_tr = pd.DataFrame()
    full_data_te = pd.DataFrame()
    for i in range(1, 4):
        data_path = './pub_dataset{}.mat'.format(i)
        with h5py.File(data_path, 'r') as f:
            full_data_tr = full_data_tr.append(
                pd.DataFrame(data=f['data_tr']).T, ignore_index=True)
            full_data_te = full_data_te.append(
                pd.DataFrame(data=f['data_te']).T, ignore_index=True)

    return full_data_tr, full_data_te


def rename_columns(ds):
    url = 'https://archive.ics.uci.edu/ml/datasets/Unmanned+Aerial+Vehicle+%28UAV%29+Intrusion+Detection'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    cat = soup.find_all('p', {'class': 'normal'})[21]
    cat = cat.get_text(separator='<br/>', strip=True).split('<br/>')[1:]
    col_names = [(re.search('(?<=. )\D+', c).group()) for c in cat]
    old_new_col = {i: col for i, col in enumerate(col_names)}
    ds.rename(old_new_col, axis=1, inplace=True)

    return ds


def filter_scale(ds):
    fil_col = [col for col in list(ds.columns) if re.search(
        '\_size_mean$|\_size_median$', col)]
    ds_sc = ds[fil_col]
    return ds_sc


get_datasets()
data_tr, data_te = import_dataset()
X_tr = filter_scale(data_tr)
y_tr = data_tr['label']


clf_svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
clf_svc.fit(X_tr, y_tr)

pickle.dump(clf_svc, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
