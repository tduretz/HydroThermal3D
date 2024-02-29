# Adimensional 3D hydrothermal simulation code

### Governing equations

- Fluid continuity equation:

$$ - \frac{\partial q^f_x}{\partial x} - \frac{\partial q^f_y}{\partial y} - \frac{\partial q^f_z}{\partial z} = \mathrm{Ra} \frac{\partial \left(k T \right)}{\partial y}$$

The right-hand side is assembled in [kernels_HT3D.jl](/kernels_HT3D.jl) function `Compute_Rp!()`. There you can set a fluid pressure source or sink, for example. The residual of the equation is evaluated in [kernels_HT3D.jl](/kernels_HT3D.jl) function `ResidualDiffusion!()`.

- Darcy flux:

$$ q^f_i = - k^f \frac{\partial P}{\partial x_i} $$

The Darcy flux vector components are evaluated in [kernels_HT3D.jl](/kernels_HT3D.jl) function `ComputeFlux!()`.

- Heat equation:
  
 $$  - \frac{\partial q^t_x}{\partial x} - \frac{\partial q^t_y}{\partial y} - \frac{\partial q^t_z}{\partial z} = \frac{\partial T}{\partial t} + v_x \frac{\partial T}{\partial x} + v_y \frac{\partial T}{\partial y} + v_z \frac{\partial T}{\partial z} \$$

The residual of the equation is evaluated in [kernels_HT3D.jl](/kernels_HT3D.jl) function `ResidualDiffusion!()`. The first part of the right-hand side $\left(\frac{\partial T}{\partial t} \right)$ is assembled in [kernels_HT3D.jl](/kernels_HT3D.jl) function `InitThermal!()`. For now it's only $-\frac{T^{old}}{\Delta t}$, which arises from the time discretisation of $\frac{\partial T}{\partial t} \approx \frac{T - T^{old}}{\Delta t}$. This is the place where you can add a **heat source**, for example. The second part of the right-hand side is the advection term, it is solved seperately in [kernels_HT3D.jl](/kernels_HT3D.jl) function `AdvectWithWeno5!()`.

 - Heat flux:

$$ q^t_i = - k^t \frac{\partial T}{\partial x_i} $$

The heat flux vector components are evaluated in [kernels_HT3D.jl](/kernels_HT3D.jl) function `ComputeFlux!()`.

   
- Fluid velocity:

$$ v_i = - k \frac{\partial P}{\partial x_i} - \mathrm{Ra} T g_i  $$

where $g = [0 \; 1 \; 0]^T$. The components of the fluid velocity vector are evaluated in [kernels_HT3D.jl](/kernels_HT3D.jl) function `Init_Vel!()`.


### Run the code locally

*Preliminary step*: Install VSCode and Julia extension. Follow these steps:

https://youtu.be/N_CQQgKEbdc

*Also relevant*:

https://youtu.be/ldRp7xvpLeA

https://youtu.be/kV8zw6quCA8


*Installation steps*:

0. Clone the folder on your local machine.

2. Open VSCode.

3. File, open folder and select the folder (`HydroThermal3D`).

4. Open Julia's REPL `Ctrl+Shift+P` and type.`Start REPL` in the search bar, click on it.

5. Go to package mode by typing `]`.

6. Type `instantiate` and press enter. 

7. ... takes time as it installs all the dependencies

8. Once finish, open a script (e.g. `HydroThermal3D_v8_Khaled.jl`) and run it by pressing the play button.

### Cluster

An example of job submission script is available in the root of the repository.

### Examples

1. 3D porous convection after 750 steps
![](./images/PorousConvectionStep0750.png)
