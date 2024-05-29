import numpy as np

def vmm(acc, dPsi, x_init=0, y_init=0, v_init=30, psi_init=0, dt=1.0/25):
    def update_state(pos_x, pos_y, vel, heading, a_x, dPsi, dt):
        pos_x = pos_x + vel * np.cos(heading) * dt + a_x*np.cos(heading) *(dt**2/2) - dPsi*vel*np.sin(heading)* (dt**2/2)
        pos_y = pos_y + vel * np.sin(heading) * dt + a_x*np.sin(heading) *(dt**2/2) + dPsi*vel*np.cos(heading)* (dt**2/2)
        velocity = vel + a_x*dt
        heading = heading + dPsi*dt
        return pos_x, pos_y, velocity, heading
   
    # Update npe state for time steps
    xn = x_init
    yn = y_init
    veln = v_init
    psin = psi_init

    x=np.zeros_like(acc)
    y=np.zeros_like(acc) 
    vel=np.zeros_like(acc)
    psi = np.zeros_like(acc)
    for i in range(acc.shape[-1]):
        xn, yn, veln, psin = update_state(xn, yn, veln, psin, acc[i].reshape(-1,1), dPsi[i].reshape(-1,1), dt)
        x[i] = xn[0,0]
        y[i] = yn[0,0]
        psi[i] = psin[0,0]
        vel[i] = veln[0,0]
    return x, y, vel, psi

