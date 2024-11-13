import numpy as np

def FiniteDifference(temp_grid, 
                     heat_deposition, 
                     attenuation,  
                     delta_x,
                     delta_y,
                     delta_z,
                     delta_t,
                     initial_temp = 37,
                     k_t = 0.49496875,
                     rho_0 = 1090.4,
                     C_t = 3421.2
                     ): 

    Nz, Ny, Nx = np.shape(heat_deposition)
    
    returning_temp = np.zeros_like(temp_grid) + initial_temp

    heat_const = delta_t / (rho_0 * C_t)
    time_constant = 1 - 2 * delta_t * k_t / (rho_0 * C_t) * (delta_x ** (-2) + delta_y ** (-2) + delta_z ** (-2))
    x_constant = delta_t * k_t / (rho_0 * C_t) * (delta_x ** (-2))
    y_constant = delta_t * k_t / (rho_0 * C_t) * (delta_y ** (-2))
    z_constant = delta_t * k_t / (rho_0 * C_t) * (delta_z ** (-2))

    returning_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] = (heat_deposition[1:Nz-1, 1:Ny-1, 1:Nx-1] * (attenuation ** 2) * heat_const + 
                             temp_grid[1:Nz-1, 1:Ny-1, 1:Nx-1] * time_constant +
                             (temp_grid[1:Nz-1, 1:Ny-1, 0:Nx-2] + temp_grid[1:Nz-1, 1:Ny-1, 2:Nx]) * x_constant + 
                             (temp_grid[1:Nz-1, 0:Ny-2, 1:Nx-1] + temp_grid[1:Nz-1, 2:Ny, 1:Nx-1]) * y_constant +
                             (temp_grid[0:Nz-2, 1:Ny-1, 1:Nx-1] + temp_grid[2:Nz, 1:Ny-1, 1:Nx-1]) * z_constant
                        )
    
    return returning_temp

def RectangularPulse(time, period_duration = 2e-3, period_repetition_interval = 4e-3):
    
    pulse_remainder = np.remainder(time, period_repetition_interval)
    indicator = pulse_remainder < period_duration
    amplitude = np.zeros_like(time)
    amplitude[indicator] = 1
    
    return amplitude

def TurkPulse(time, period_duration = 3.25e-3, period_repetition_interval = 4e-3, ramp_duration = 1e-3):
    
    pulse_remainder = np.remainder(time, period_repetition_interval)
    
    indicator_1 = pulse_remainder < ramp_duration
    indicator_2 = np.logical_and(pulse_remainder < period_duration - ramp_duration, pulse_remainder >= ramp_duration)
    indicator_3 = np.logical_and(pulse_remainder >= period_duration - ramp_duration, pulse_remainder < period_duration)
    
    amplitude = np.zeros_like(time)
    amplitude[indicator_2] = 1
    amplitude[indicator_1] = -np.cos(np.pi * (pulse_remainder[indicator_1] / ramp_duration)) / 2 + 0.5
    amplitude[indicator_3] = -np.cos(np.pi * ((pulse_remainder[indicator_3] - period_duration) / ramp_duration)) / 2 + 0.5
    
    return amplitude