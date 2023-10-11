# ImageRestoration
Using iterative Anisotropic Diffusion to restore graffitied or noisied images

PDE-based methods for image restoration are based on propagating the information 
(typically, intensity values and gradients) at the boundaries of the missing region
inwards. The propagation is performed by solving a partial differential equation
with specified boundary conditions.
The simplest case is harmonic painting where the Laplacian is set to zero $\Delta I = 0$ and only nearest neighbours are considered.
The anisotropic diffusion equation was also solved which has improved capability of edge detection by using a diffusion coefficient in the Laplacian that varies depending on the local gradient:
$$D\frac{du}{dt} = -\Delta u(x,t)$$ which becomes
$$\frac{du}{dt} = \nabla \cdot (c(\Vert \nabla u \Vert) \nabla u)$$ using the Perona Malik diffusivity equation:
$c(\Vert(\nabla u \Vert) = \frac{1}{1+(\frac{\Vert(\nabla u \Vert)}{\kappa})^2}.$

Example of results:
Original in colour:
![Colour Glacier](https://github.com/andersoren/ImageRestoration/blob/main/Glacier.jpg)
Result in Grayscale
 
