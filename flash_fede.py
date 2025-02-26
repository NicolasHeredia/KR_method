import numpy as np

from yaeos.core import GeModel

from scipy.optimize import root, newton


def get_K(model: GeModel, x, y, T):
  """Calculo de K como relaciones de gamma
  """
  ln_g_x = model.ln_gamma(x, T)
  ln_g_y = model.ln_gamma(y, T)
  lnK = ln_g_y - ln_g_x
  return np.exp(lnK), lnK


def rachford_rice(beta, z, K):
  """Ecuacion rachford rice, se resuelve para
  determinar la fraccion de vapor
  """
  rr_i = z * (K - 1)/( 1 + beta * (K-1))
  rr = np.sum(rr_i)
  return rr

def d_rfr_dbeta(beta, z, K):
  """Derivada de la ecuacion de rachford rice
  """
  d_rr_i = -z * (K - 1)**2 / (1 + beta * (K - 1))**2
  d_rr = np.sum(d_rr_i)
  return np.array([d_rr])


# Cosas
g0 = lambda z, K: rachford_rice(0, z, K)
g1 = lambda z, K: rachford_rice(1, z, K)


def beta_to_01(z, K):
  """Esta es para asegurarse que beta no de
  valores tramboticos
  """
  Ki = K
  while (g0(z, Ki) < 0 or g1(z, Ki) > 0):
    if g0(z, Ki) < 0:
      Ki = Ki * 1.01
    elif g1(z, Ki) > 0:
      Ki = Ki * 0.99

  return Ki


def same_activity(model: GeModel, x, y, T):
  return np.allclose(
      x * np.exp(model.ln_gamma(x, T)),
      y * np.exp(model.ln_gamma(y, T))
  )


def flash_gamma_gamma(model: GeModel, z, K0, T, beta0=0.5, max_iters=400):
  K = K0
  for i in range(max_iters):
    K = beta_to_01(z, K)

    # beta = root(rachford_rice, x0=beta0, args=(z, K)).x[0]
    beta = root(rachford_rice, x0=beta0, args=(z, K), jac=d_rfr_dbeta, method="lm").x[0]
        
    # Update molar fractions
    y = z * K / (1 + beta * (K - 1))
    x = y/K

    # Calculate new K-values
    K, lnK = get_K(model, x, y, T)

    if (same_activity(model, x, y, T)): break

  return x, y, beta, i