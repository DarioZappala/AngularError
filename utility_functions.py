from XRDXRFutils.notebook_utils import *


div_optic = rad2deg(arctan(6.2 / (2 * 102.4)))


def distance(A, B):
    return sqrt(power(A[0] - B[0], 2) + power(A[1] - B[1], 2))

def weighted_std(values, axis = None, weights = None):
    avg = average(values, axis = axis, weights = weights, keepdims = True)
    var = average((values - avg)**2, axis = axis, weights = weights)
    return sqrt(var)

def a_s_from_beta(beta, L, theta_min, theta_max):
    factor = L * cos(deg2rad(theta_max - beta)) / sin(deg2rad(theta_max - theta_min))
    a = factor * sin(deg2rad(theta_min - beta))
    s = factor * cos(deg2rad(theta_min - beta))
    return a, s


def R_alpha_gamma__to__a_s_beta(R, alpha, gamma, L):
    a = - (L/2 + R * cos(deg2rad(alpha + gamma)))
    s = R * sin(deg2rad(alpha + gamma))
    beta = 90 - gamma
    return a, s, beta

def a_s_beta__to__R_alpha_gamma(a, s, beta, L):
    R = sqrt(power(s, 2) + power(L/2 + a, 2))
    alpha = beta + rad2deg(arctan((L/2 + a) / s))
    gamma = 90 - beta
    return R, alpha, gamma


def distance_to_angle(x, a, s, beta):
    return rad2deg(arctan((x + a) / s)) + beta

def channel_to_angle(c, a, s, beta, length_channel):
    x = c * length_channel
    return distance_to_angle(x, a, s, beta)

def angle_to_distance(theta, a, s, beta):
    return s * tan(deg2rad(theta - beta)) - a

def angle_to_channel(theta, a, s, beta, length_channel):
    x = angle_to_distance(theta, a, s, beta)
    return x / length_channel

def Dp_to_Dtheta(Delta_p, p, a, s):
    return rad2deg(( abs(s) / (power(s, 2) + power(p + a, 2)) ) * Delta_p)


def error_defocus(theta, a, s, beta, i, d, omega):
    Delta_f = (1 / cos(deg2rad(theta - beta))) * s - (sin(deg2rad(theta - i)) / sin(deg2rad(i))) * d
    Delta_p = (sin(deg2rad(omega)) / cos(deg2rad(theta - beta + omega))) * Delta_f
    p = angle_to_distance(theta, a, s, beta)
    return Delta_f, Delta_p, Dp_to_Dtheta(Delta_p, p, a, s)


def conic_diffraction_base(z, theta, a, s, beta):
    my_radicand = (power(s, 2) + power(z, 2)) * power(sin(deg2rad(theta)), 2) - power(z, 2) * power(cos(deg2rad(beta)), 2)
    if (type(my_radicand) == ndarray):
        my_radical = sqrt(my_radicand, out = ones(my_radicand.shape) * nan, where = (my_radicand >= 0))
    else:
        if (my_radicand >= 0):
            my_radical = sqrt(my_radicand)
        else:
            my_radical = nan
    aux = power(z, 2) * cos(deg2rad(theta)) / (s * sin(deg2rad(theta)) + my_radical)
    return array([
        s * tan(deg2rad(theta - beta)) - a - aux,
        - s * tan(deg2rad(theta + beta)) - a + aux
    ])

def conic_diffraction(z, theta, a, s, beta, d, i, omega, psi):
    Delta_f = (1 / cos(deg2rad(theta - beta))) * s - (sin(deg2rad(theta - i)) / sin(deg2rad(i))) * d
    Delta_p = (sin(deg2rad(omega)) / cos(deg2rad(theta - beta + omega))) * Delta_f
    Delta_z = ( d + s / cos(deg2rad(theta - beta)) ) * tan(deg2rad(psi))
    return conic_diffraction_base(z - Delta_z, theta, a, s, beta) + Delta_p

def error_defocus_and_cone(width_channel, theta, a, s, beta, d, i, sigma_omega, sigma_psi):
    values = stack([
        conic_diffraction(width_channel * 0, theta, a, s, beta, d, i, -sigma_omega, sigma_psi * 0)[0],
        conic_diffraction(width_channel * 0, theta, a, s, beta, d, i, sigma_omega, sigma_psi * 0)[0],
        conic_diffraction(width_channel/2, theta, a, s, beta, d, i, -sigma_omega, -sigma_psi)[0],
        conic_diffraction(width_channel/2, theta, a, s, beta, d, i, +sigma_omega, -sigma_psi)[0]
    ], axis = -1)
    Delta_p = 0.5 * (amax(values, axis = -1) - amin(values, axis = -1))
    p = angle_to_distance(theta, a, s, beta)
    return array([Delta_p, Dp_to_Dtheta(Delta_p, p, a, s)])


def error_resolution(theta, a, s, beta, length_channel):
    p = angle_to_distance(theta, a, s, beta)
    return Dp_to_Dtheta(length_channel, p, a, s)


def plot_experimental_setting(L, a, s, beta, i, d, omega, theta, ax, legend = False, secondary_ray = True):
    # Calculations
    coords_S = [-d * cos(deg2rad(i)), d * sin(deg2rad(i))]
    coords_O = [0, 0]
    coords_O1 = [(sin(deg2rad(omega)) * cos(deg2rad(omega)) / sin(deg2rad(i))) * d, (power(sin(deg2rad(omega)), 2) / sin(deg2rad(i))) * d]
    coords_F = [d * sin(deg2rad(theta - i)) * cos(deg2rad(theta - i)) / sin(deg2rad(i)), d * power(sin(deg2rad(theta - i)), 2) / sin(deg2rad(i))]
    coords_A = [s * cos(deg2rad(beta - i)) - a * sin(deg2rad(beta - i)), a * cos(deg2rad(beta - i)) + s * sin(deg2rad(beta - i))]
    coords_B = [s * cos(deg2rad(beta - i)) - (L + a) * sin(deg2rad(beta - i)), (L + a) * cos(deg2rad(beta - i)) + s * sin(deg2rad(beta - i))]
    coords_Q = [s * cos(deg2rad(beta - i)), s * sin(deg2rad(beta - i))]
    coords_P = [s * cos(deg2rad(theta - i)) / cos(deg2rad(theta - beta)), s * sin(deg2rad(theta - i)) / cos(deg2rad(theta - beta))]
    coords_P1 = [
        ( 1 / cos(deg2rad(theta + omega - beta)) ) * ( (sin(deg2rad(beta - i)) * sin(deg2rad(omega)) * sin(deg2rad(theta - i)) / sin(deg2rad(i))) * d + cos(deg2rad(theta - i + omega)) * s ),
        ( 1 / cos(deg2rad(theta + omega - beta)) ) * ( sin(deg2rad(theta - i + omega)) * s - ( cos(deg2rad(beta - i)) * sin(deg2rad(omega)) * sin(deg2rad(theta - i)) / sin(deg2rad(i)) ) * d )
    ]

    ### Figure
    ax.set_aspect(1)

    # Draw the circle
    radius = d / (2 * sin(deg2rad(i)))
    circle_focus = Circle((0, radius), radius, fill = False, color = 'black', lw = 1, ls = '--')
    ax.add_artist(circle_focus)

    # Draw the detector
    ax.plot([coords_A[0], coords_B[0]], [coords_A[1], coords_B[1]], color = 'black', lw = 2)
    ax.plot([coords_O[0], coords_Q[0]], [coords_O[1], coords_Q[1]], color = 'black', lw = 1, ls = '--')
    if ((a > 0) or (a < -L)):
        ax.plot([coords_A[0], coords_Q[0]], [coords_A[1], coords_Q[1]], color = 'black', lw = 1, ls = '--')

    # Draw the rays
    ax.plot([coords_S[0], coords_O[0]], [coords_S[1], coords_O[1]], color = 'tab:blue', lw = 1.5, label = 'central ray')
    ax.plot([coords_O[0], coords_F[0]], [coords_O[1], coords_F[1]], color = 'tab:blue', lw = 1.5)
    ax.plot([coords_P[0], coords_F[0]], [coords_P[1], coords_F[1]], color = 'tab:blue', lw = 1.5)
    if secondary_ray:
        ax.plot([coords_S[0], coords_O1[0]], [coords_S[1], coords_O1[1]], color = 'tab:orange', lw = 1.5, label = 'secondary ray')
        ax.plot([coords_O1[0], coords_F[0]], [coords_O1[1], coords_F[1]], color = 'tab:orange', lw = 1.5)
        ax.plot([coords_P1[0], coords_F[0]], [coords_P1[1], coords_F[1]], color = 'tab:orange', lw = 1.5)

    # Mark the points
    list_coords = [coords_S, coords_O, coords_F, coords_A, coords_B, coords_P]
    list_letters = ['S', 'O', 'F', 'A', 'B', 'P']
    if secondary_ray:
        list_coords += [coords_O1, coords_P1]
        list_letters += ['O\'', 'P\'']
    for coords, letter in zip(list_coords, list_letters):
        ax.scatter(*coords, c = 'black', s = 20, zorder = 2.5)
        ax.annotate(letter, coords, ha = 'left', va = 'bottom')

    ax.set_xlabel(r'$x$ (cm)')
    ax.set_ylabel(r'$y$ (cm)')
    ax.set_title('System setup')
    if legend:
        ax.legend(frameon = True, bbox_to_anchor = (1, 1), loc = 'upper left')
