"""Shared continuous margins. No target/test selection occurs in this module."""
from __future__ import annotations
from release_paths import work_path as _work_path, raw_path as _raw_path
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
from lightgbm import __version__ as LIGHTGBM_VERSION
from scipy import optimize, special, stats
from lightgbm import LGBMRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

QUANTILES = np.array([.005,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995])


class GaussianAdditiveMargin:
    """Regularized additive spline mean and log scale; beta PIT recalibration external.

    This is an implemented location-scale additive baseline, not a reproduction
    of the Gioia covariance model. Its smoothness is controlled by ridge alpha.
    """
    def __init__(self,alpha=10.,n_knots=5,scale_exclude_features=(),
                 scale_extrapolation='constant',mean_smooth_features=None,
                 mean_extrapolation='linear',scale_smooth_features=None):
        self.alpha=alpha;self.n_knots=n_knots
        self.scale_exclude_features=tuple(scale_exclude_features)
        self.scale_extrapolation=scale_extrapolation
        # None preserves the original unique-value rule. Explicit input indices
        # fix feature roles across training windows; every remaining input is
        # linear. Existing fitted pickle objects use their saved masks/bases.
        self.mean_smooth_features=None if mean_smooth_features is None else tuple(mean_smooth_features)
        self.mean_extrapolation=mean_extrapolation
        self.scale_smooth_features=None if scale_smooth_features is None else tuple(scale_smooth_features)

    @staticmethod
    def _explicit_smooth_mask(indices,n_features):
        if (len(set(indices))!=len(indices) or
                any(isinstance(j,(bool,np.bool_)) or not isinstance(j,(int,np.integer))
                    or j<0 or j>=n_features for j in indices)):
            raise ValueError('Smooth feature indices must be unique integers within the input schema.')
        return np.isin(np.arange(n_features),indices)

    def _design(self,X,fit=False):
        X=np.asarray(X,float)
        if fit:
            self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
            X=self.imputer_.fit_transform(X)
            self.continuous_=(np.array([len(np.unique(X[:,j]))>10 for j in range(X.shape[1])])
                if self.mean_smooth_features is None else
                self._explicit_smooth_mask(self.mean_smooth_features,X.shape[1]))
            self.spline_=SplineTransformer(n_knots=self.n_knots,degree=3,
                                           knots='quantile',extrapolation=self.mean_extrapolation,include_bias=False)
            smooth=(self.spline_.fit_transform(X[:,self.continuous_]) if self.continuous_.any()
                    else np.empty((len(X),0)))
        else:
            X=self.imputer_.transform(X)
            smooth=(self.spline_.transform(X[:,self.continuous_]) if self.continuous_.any()
                    else np.empty((len(X),0)))
        Z=np.column_stack([smooth,X[:,~self.continuous_]])
        if fit:
            self.scaler_=StandardScaler()
            return self.scaler_.fit_transform(Z)
        return self.scaler_.transform(Z)

    def _scale_design(self,X,fit=False):
        """Separate scale basis: no unbounded log-variance extrapolation.

        Deterministic slow-time features can be excluded by their documented
        input indices. Feature choice must be made before the held-out test.
        Constant extrapolation is applied at TRAINING feature boundaries.
        """
        X=np.asarray(X,float)
        if fit:
            excluded=set(self.scale_exclude_features)
            if any(not isinstance(j,(int,np.integer)) or j<0 or j>=X.shape[1] for j in excluded):
                raise ValueError('Invalid scale exclusion feature indices.')
            self.scale_features_=np.array([j for j in range(X.shape[1]) if j not in excluded],int)
            if not len(self.scale_features_):
                self.constant_scale_=True
                return np.ones((len(X),1))
            self.constant_scale_=False
            self.scale_imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
            W=self.scale_imputer_.fit_transform(X[:,self.scale_features_])
            if self.scale_smooth_features is None:
                self.scale_continuous_=np.array([len(np.unique(W[:,j]))>10 for j in range(W.shape[1])])
            else:
                smooth_mask=self._explicit_smooth_mask(self.scale_smooth_features,X.shape[1])
                if any(smooth_mask[j] for j in excluded):
                    raise ValueError('An excluded scale input cannot also be a smooth scale input.')
                self.scale_continuous_=smooth_mask[self.scale_features_]
            self.scale_spline_=SplineTransformer(n_knots=self.n_knots,degree=3,
                knots='quantile',extrapolation=self.scale_extrapolation,include_bias=False)
            smooth=self.scale_spline_.fit_transform(W[:,self.scale_continuous_]) if self.scale_continuous_.any() else np.empty((len(W),0))
            Z=np.column_stack([smooth,W[:,~self.scale_continuous_]])
            self.scale_scaler_=StandardScaler()
            return self.scale_scaler_.fit_transform(Z)
        if self.constant_scale_:return np.ones((len(X),1))
        W=self.scale_imputer_.transform(X[:,self.scale_features_])
        smooth=self.scale_spline_.transform(W[:,self.scale_continuous_]) if self.scale_continuous_.any() else np.empty((len(W),0))
        return self.scale_scaler_.transform(np.column_stack([smooth,W[:,~self.scale_continuous_]]))

    def fit(self,X,y):
        y=np.asarray(y,float)
        Z=self._design(X,True)
        self.mean_=Ridge(alpha=self.alpha).fit(Z,y)
        residual=y-self.mean_.predict(Z)
        self.floor_=max(np.std(y)*1e-6,1e-8)
        W=self._scale_design(X,True)
        self.scale_=Ridge(alpha=self.alpha).fit(W,np.log(np.maximum(residual**2,self.floor_**2)))
        rawvar=np.exp(np.clip(self.scale_.predict(W),-40,40))
        self.variance_factor_=np.mean(residual**2/rawvar)
        return self

    def parameters(self,X):
        Z=self._design(X)
        mean=self.mean_.predict(Z)
        # Retain readability of earlier development-only pickle artifacts.
        W=self._scale_design(X) if hasattr(self,'scale_features_') else Z
        sd=np.sqrt(np.exp(np.clip(self.scale_.predict(W),-40,40))*self.variance_factor_)
        return mean,np.maximum(sd,self.floor_)

    def cdf(self,X,y):
        mean,sd=self.parameters(X)
        return np.clip(stats.norm.cdf((np.asarray(y)-mean)/sd),1e-12,1-1e-12)

    def predict_quantiles(self,X):
        mean,sd=self.parameters(X)
        return mean[:,None]+sd[:,None]*stats.norm.ppf(QUANTILES)[None,:]


class QuantileMargin:
    """Conditional quantiles with exponential tails and monotone rearrangement.

    Tail scales are conditional spacings between the two outer fitted quantiles.
    A training response scale supplies only a numerical floor, not test data.
    """
    def __init__(self, quantiles=QUANTILES, n_estimators=250, num_leaves=15,
                 learning_rate=.04, min_child_samples=120, seed=20260929, n_jobs=2,
                 checkpoint_root=None):
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.params = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                           learning_rate=learning_rate, min_child_samples=min_child_samples,
                           random_state=seed, n_jobs=n_jobs, verbosity=-1,
                           reg_lambda=1., subsample=1., colsample_bytree=1.)
        self.checkpoint_root = None if checkpoint_root is None else str(checkpoint_root)

    CHECKPOINT_VERSION = '1.0-training-only-atomic-quantile-models'

    @staticmethod
    def _file_sha256(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _training_array_contract(array):
        if array.dtype.hasobject:
            raise ValueError('Checkpoint training arrays must have numeric, non-object dtype.')
        contiguous = np.ascontiguousarray(array)
        return dict(shape=list(array.shape), dtype=array.dtype.str,
                    sha256=hashlib.sha256(contiguous.tobytes(order='C')).hexdigest())

    def _checkpoint_context(self, X, y):
        supplied = getattr(self, 'checkpoint_root', None)
        if supplied is None:
            return None
        allowed = _work_path('cache').resolve()
        root = Path(supplied).resolve()
        if not root.is_relative_to(_work_path().resolve()):
            raise ValueError('Quantile model checkpoints must remain under the configured work root.')
        try:
            root.relative_to(allowed)
        except ValueError as error:
            raise ValueError('Quantile model checkpoints must stay inside the project cache subtree.') from error
        contract = dict(version=self.CHECKPOINT_VERSION,
            X=self._training_array_contract(X), y=self._training_array_contract(y),
            quantiles=self.quantiles.tolist(), parameters=self.params,
            margins_sha256=self._file_sha256(__file__), lightgbm_version=LIGHTGBM_VERSION)
        encoded = json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')
        key = hashlib.sha256(encoded).hexdigest()
        directory = root/key
        directory.mkdir(parents=True, exist_ok=True)
        return dict(directory=directory, key=key, contract=contract)

    def _checkpoint_receipt_contract(self, context, position, q):
        return dict(version=self.CHECKPOINT_VERSION, cache_key=context['key'],
            training_contract=context['contract'], quantile_index=int(position), quantile=float(q))

    def _checkpoint_load(self, context, position, q):
        receipt_path = context['directory']/f'q{position:02d}.json'
        # Uncommitted .tmp files and orphan content-addressed pickles are ignored.
        if not receipt_path.exists():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            expected = self._checkpoint_receipt_contract(context, position, q)
            if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
                raise ValueError('Receipt training key, quantile or parameters differ.')
            digest = receipt.get('model_sha256')
            filename = receipt.get('model_file')
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(c not in '0123456789abcdef' for c in digest) or
                    filename != f'q{position:02d}.{digest}.pkl'):
                raise ValueError('Malformed model filename or SHA256.')
            path = context['directory']/filename
            if not path.is_file() or self._file_sha256(path) != digest:
                raise ValueError('Committed checkpoint model is missing or fails SHA256 verification.')
            with path.open('rb') as stream:
                model = pickle.load(stream)
            if not isinstance(model, LGBMRegressor) or not model.__sklearn_is_fitted__():
                raise ValueError('Checkpoint is not a fitted LightGBM regression model.')
            parameters = model.get_params()
            if (parameters.get('objective') != 'quantile' or parameters.get('alpha') != float(q)
                    or any(parameters.get(k) != v for k, v in self.params.items())
                    or model.n_features_in_ != context['contract']['X']['shape'][1]):
                raise ValueError('Checkpoint model parameters or input feature count differ.')
            return model
        except Exception as error:
            raise RuntimeError(f'Invalid committed quantile checkpoint: {receipt_path}: {error}') from error

    def _checkpoint_save(self, context, position, q, model):
        directory = context['directory']
        # Commit the content-addressed pickle first, then its small receipt.
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f'q{position:02d}.',
                                          suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        digest = self._file_sha256(temporary)
        model_path = directory/f'q{position:02d}.{digest}.pkl'
        if model_path.exists():
            if self._file_sha256(model_path) != digest:
                raise RuntimeError('Existing content-addressed checkpoint fails its filename hash.')
            temporary.unlink()
        else:
            temporary.replace(model_path)
        receipt = {**self._checkpoint_receipt_contract(context, position, q),
                   'model_file': model_path.name, 'model_sha256': digest}
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f'q{position:02d}.receipt.',
                suffix='.tmp', mode='w', encoding='utf8', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory/f'q{position:02d}.json')

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        if len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError('Training labels must be finite and aligned.')
        self.floor_ = max(float(np.std(y)) * 1e-6, 1e-8)
        context = self._checkpoint_context(X, y)
        self.checkpoint_report_ = []
        self.models_ = []
        for position, q in enumerate(self.quantiles):
            if context is not None and self._file_sha256(__file__) != context['contract']['margins_sha256']:
                raise RuntimeError('Margin source changed during checkpointed fitting.')
            model = self._checkpoint_load(context, position, q) if context is not None else None
            reused = model is not None
            if not reused:
                model = LGBMRegressor(objective='quantile', alpha=float(q), **self.params)
                model.fit(X, y)
                if context is not None:
                    if self._file_sha256(__file__) != context['contract']['margins_sha256']:
                        raise RuntimeError('Margin source changed before checkpoint commit.')
                    self._checkpoint_save(context, position, q, model)
            self.models_.append(model)
            if context is not None:
                self.checkpoint_report_.append(dict(quantile_index=position, quantile=float(q),
                    cache_key=context['key'], status='reused' if reused else 'fitted_and_committed'))
        return self

    def predict_quantiles(self, X):
        values = np.column_stack([m.predict(np.asarray(X)) for m in self.models_])
        values.sort(axis=1)
        # Strict spacing gives a continuous, invertible CDF, including tied leaves.
        for j in range(1, values.shape[1]):
            values[:, j] = np.maximum(values[:, j], values[:, j-1] + self.floor_)
        return values

    def cdf(self, X, y):
        return quantile_cdf(self.predict_quantiles(X), y, self.quantiles, self.floor_)


def quantile_cdf(values, y, levels=QUANTILES, scale_floor=1e-8):
    values, levels = np.asarray(values, float), np.asarray(levels, float)
    y = np.broadcast_to(np.asarray(y, float), (len(values),))
    if values.shape[1] != len(levels) or np.any(np.diff(values, axis=1) <= 0):
        raise ValueError('Quantile values must increase strictly.')
    out = np.array([np.interp(t, row, levels) for t, row in zip(y, values)])
    low = y < values[:, 0]
    high = y > values[:, -1]
    sl = np.maximum((values[:,1]-values[:,0])/np.log(levels[1]/levels[0]), scale_floor)
    su = np.maximum((values[:,-1]-values[:,-2])/np.log((1-levels[-2])/(1-levels[-1])), scale_floor)
    out[low] = levels[0] * np.exp(np.maximum((y[low]-values[low,0])/sl[low], -745))
    out[high] = 1-(1-levels[-1])*np.exp(np.maximum(-(y[high]-values[high,-1])/su[high], -745))
    return np.clip(out, 1e-12, 1-1e-12)


def quantile_ppf(values, p, levels=QUANTILES, scale_floor=1e-8):
    values, levels = np.asarray(values, float), np.asarray(levels, float)
    p = np.clip(np.broadcast_to(np.asarray(p, float), (len(values),)), 1e-12, 1-1e-12)
    out = np.array([np.interp(t, levels, row) for t, row in zip(p, values)])
    low, high = p < levels[0], p > levels[-1]
    sl = np.maximum((values[:,1]-values[:,0])/np.log(levels[1]/levels[0]), scale_floor)
    su = np.maximum((values[:,-1]-values[:,-2])/np.log((1-levels[-2])/(1-levels[-1])), scale_floor)
    out[low] = values[low,0]+sl[low]*np.log(p[low]/levels[0])
    out[high] = values[high,-1]-su[high]*np.log((1-p[high])/(1-levels[-1]))
    return out


class BetaPITCalibration:
    """Two-parameter monotone beta CDF, fitted only on held-out PIT values."""
    def fit(self, u):
        u = np.clip(np.asarray(u, float), 1e-8, 1-1e-8)
        if not np.isfinite(u).all() or len(u) < 100:
            raise ValueError('At least 100 finite calibration PIT observations required.')
        logu, logv = np.log(u).mean(), np.log1p(-u).mean()
        def loss(theta):
            a,b = np.exp(theta)
            return special.betaln(a,b)-(a-1)*logu-(b-1)*logv
        result = optimize.minimize(loss, np.zeros(2), method='L-BFGS-B', bounds=[(-2,2)]*2)
        if not result.success:
            raise RuntimeError(f'PIT calibration failed: {result.message}')
        self.a_, self.b_ = np.exp(result.x)
        return self

    def transform(self, u):
        return np.clip(special.betainc(self.a_,self.b_,np.asarray(u,float)),1e-12,1-1e-12)

    def inverse(self, p):
        return special.betaincinv(self.a_,self.b_,np.asarray(p,float))


COHERENT_CALIBRATION_VERSION = '1.0-dual-tail-no-query-epsilon-clipping'


def _quantile_tail_inputs(values, levels, scale_floor):
    values, levels = np.asarray(values, float), np.asarray(levels, float)
    if (values.ndim != 2 or values.shape[1] != len(levels) or len(levels) < 2
            or not np.isfinite(values).all() or not np.isfinite(levels).all()
            or np.any(np.diff(values, axis=1) <= 0) or np.any(np.diff(levels) <= 0)
            or levels[0] <= 0 or levels[-1] >= 1 or scale_floor <= 0):
        raise ValueError('Finite, strictly increasing quantile rows/levels and positive scale floor required.')
    lower = np.maximum((values[:, 1]-values[:, 0])/np.log(levels[1]/levels[0]), scale_floor)
    upper = np.maximum((values[:, -1]-values[:, -2])/
                       (np.log1p(-levels[-2])-np.log1p(-levels[-1])), scale_floor)
    return values, levels, lower, upper


def quantile_cdf_sf(values, y, levels=QUANTILES, scale_floor=1e-8):
    """Unclipped CDF/survival for the existing linear-grid/exponential-tail law.

    Separate tails retain small survival probabilities even when 1-CDF rounds
    to zero. The legacy quantile_cdf/quantile_ppf interfaces remain unchanged.
    """
    values, levels, lower, upper = _quantile_tail_inputs(values, levels, scale_floor)
    y = np.broadcast_to(np.asarray(y, float), (len(values),))
    if np.isnan(y).any():
        raise ValueError('Query outcomes must not be NaN.')
    cdf = np.array([np.interp(t, row, levels) for t, row in zip(y, values)])
    sf = np.array([np.interp(t, row, 1-levels) for t, row in zip(y, values)])
    low, high = y < values[:, 0], y > values[:, -1]
    logcdf = np.log(levels[0])+(y[low]-values[low, 0])/lower[low]
    cdf[low], sf[low] = np.exp(logcdf), -np.expm1(logcdf)
    logsf = np.log1p(-levels[-1])-(y[high]-values[high, -1])/upper[high]
    sf[high], cdf[high] = np.exp(logsf), -np.expm1(logsf)
    return cdf, sf


def quantile_tail_inverse(values, probability, levels=QUANTILES, scale_floor=1e-8,
                          *, survival=False):
    """PPF or inverse survival without an epsilon change to target probability."""
    values, levels, lower, upper = _quantile_tail_inputs(values, levels, scale_floor)
    p = np.broadcast_to(np.asarray(probability, float), (len(values),))
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Inverse probabilities must be finite in [0,1].')
    with np.errstate(divide='ignore'):
        if survival:
            out = np.array([np.interp(s, (1-levels)[::-1], row[::-1])
                            for s, row in zip(p, values)])
            high, low = p < 1-levels[-1], p > 1-levels[0]
            out[high] = values[high, -1]-upper[high]*(np.log(p[high])-np.log1p(-levels[-1]))
            out[low] = values[low, 0]+lower[low]*(np.log1p(-p[low])-np.log(levels[0]))
        else:
            out = np.array([np.interp(s, levels, row) for s, row in zip(p, values)])
            low, high = p < levels[0], p > levels[-1]
            out[low] = values[low, 0]+lower[low]*(np.log(p[low])-np.log(levels[0]))
            out[high] = values[high, -1]-upper[high]*(np.log1p(-p[high])-np.log1p(-levels[-1]))
    return out


def raw_margin_cdf_sf(model, X, y, *, raw_quantiles=None):
    """Dual-tail query for new prequential forecasts, including old pickles."""
    if isinstance(model, GaussianAdditiveMargin):
        mean, sd = model.parameters(X)
        mean, sd = np.asarray(mean, float), np.asarray(sd, float)
        if not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ValueError('Normal locations/scales must be finite with positive scale.')
        z = (np.asarray(y, float)-mean)/sd
        if np.isnan(z).any():
            raise ValueError('Normal query produced NaN.')
        return stats.norm.cdf(z), stats.norm.sf(z)
    if isinstance(model, QuantileMargin):
        values = model.predict_quantiles(X) if raw_quantiles is None else raw_quantiles
        return quantile_cdf_sf(values, y, model.quantiles, model.floor_)
    raise TypeError('Coherent queries support GaussianAdditiveMargin and QuantileMargin.')


def beta_cdf_sf(calibrator, cdf, sf):
    """Evaluate G and its survival from the two original raw tails.

    B(a,b,u)=1-B(b,a,1-u). Choose the smaller calibrated tail before
    complementing, rather than feeding an epsilon-clipped raw CDF to G.
    Beta MLE's explicit 1e-8 input clipping is a separate training operation.
    """
    cdf, sf = np.broadcast_arrays(np.asarray(cdf, float), np.asarray(sf, float))
    if (not np.isfinite(cdf).all() or not np.isfinite(sf).all()
            or np.any((cdf < 0) | (cdf > 1) | (sf < 0) | (sf > 1))
            or not np.allclose(cdf+sf, 1., rtol=0., atol=2e-14)):
        raise ValueError('Raw CDF/survival must be finite, compatible probabilities.')
    a, b = float(calibrator.a_), float(calibrator.b_)
    if not np.isfinite([a, b]).all() or a <= 0 or b <= 0:
        raise ValueError('Beta shape parameters must be finite and positive.')
    lower = special.betainc(a, b, cdf)
    upper = special.betainc(b, a, sf)
    use_lower = lower <= upper
    return np.where(use_lower, lower, 1-upper), np.where(use_lower, 1-lower, upper)


def calibrated_margin_cdf_sf(model, calibrator, X, y, *, raw_quantiles=None):
    cdf, sf = raw_margin_cdf_sf(model, X, y, raw_quantiles=raw_quantiles)
    return beta_cdf_sf(calibrator, cdf, sf)


def calibrated_margin_quantiles(model, calibrator, X, levels=QUANTILES,
                                *, raw_quantiles=None):
    """G^-1 via its lower tail or swapped-shape survival inverse, then F^-1.

    Upper queries avoid constructing 1-small_survival before norm.isf or the
    exponential-tail inverse. No query epsilon is used; finite machine
    representability remains a numerical limitation, not extra probability mass.
    """
    levels = np.asarray(levels, float)
    if levels.ndim != 1 or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError('Calibrated quantile levels must be finite and strictly inside (0,1).')
    a, b = float(calibrator.a_), float(calibrator.b_)
    if not np.isfinite([a, b]).all() or a <= 0 or b <= 0:
        raise ValueError('Beta shapes must be finite and positive.')
    if isinstance(model, GaussianAdditiveMargin):
        mean, sd = model.parameters(X)
    elif isinstance(model, QuantileMargin):
        values = model.predict_quantiles(X) if raw_quantiles is None else raw_quantiles
    else:
        raise TypeError('Unsupported margin family for coherent quantiles.')
    result = []
    for q in levels:
        survival = q > .5
        p = special.betaincinv(b, a, 1-q) if survival else special.betaincinv(a, b, q)
        if isinstance(model, GaussianAdditiveMargin):
            standardized = stats.norm.isf(p) if survival else stats.norm.ppf(p)
            value = mean+sd*standardized
        else:
            value = quantile_tail_inverse(values, p, model.quantiles, model.floor_, survival=survival)
        result.append(value)
    return np.column_stack(result)


def pinball_loss(y, quantile_predictions, levels=QUANTILES):
    residual = np.asarray(y)[:,None]-np.asarray(quantile_predictions)
    return np.maximum(np.asarray(levels)*residual, (np.asarray(levels)-1)*residual).mean(axis=0)


def integrated_quantile_score(y, quantile_predictions, levels=QUANTILES):
    """Truncated CRPS approximation over the stated quantile grid, not exact CRPS."""
    residual = np.asarray(y)[:,None]-np.asarray(quantile_predictions)
    losses = np.maximum(np.asarray(levels)*residual, (np.asarray(levels)-1)*residual)
    return 2*np.trapezoid(losses, x=np.asarray(levels), axis=1)
