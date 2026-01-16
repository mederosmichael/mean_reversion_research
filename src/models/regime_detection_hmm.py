import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib

LOOKBACK = 12
TESTING_DAYS = 90

df = pd.read_parquet("data/derived/close_5m.parquet")

r = df.pct_change()
ret = r.rolling(LOOKBACK).mean()
mean_return = ret.mean(axis=1)
r_rel = ret.sub(mean_return, axis=0)
vol = np.sqrt((r_rel ** 2).mean(axis=1))
mad = r_rel.abs().median(axis=1)
mad_smooth = mad.rolling(LOOKBACK).median()

feature_space = pd.concat(
    [mean_return.rename("mu"), vol.rename("sigma"), mad_smooth.rename("mad")],
    axis=1,
).dropna()

time_end = feature_space.index.max()
t_split = time_end - pd.Timedelta(days=TESTING_DAYS)

train_df = feature_space.loc[:t_split]
test_df = feature_space.loc[t_split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.values)
X_test = scaler.transform(test_df.values)

hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=0)
hmm.fit(X_train)

train_states = hmm.predict(X_train)
test_states = hmm.predict(X_test)

train_df = train_df.assign(state=train_states)
test_df = test_df.assign(state=test_states)

print(hmm.score(X_test) / len(X_test))
joblib.dump(hmm, "models/regime_hmm.joblib")
joblib.dump(scaler, "models/regime_scaler.joblib")

