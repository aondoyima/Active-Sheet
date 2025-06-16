# Active-Sheet
Calculation of flow field and stresses in a model for epithelial tissue with active mechano-chemical feedback. 

The model is described in my paper here: https://doi.org/10.1103/PhysRevLett.131.238301 but the equations are
\begin{equation} \label{eq:myosin_equation}
\tau_m\accentset{\circ}{\bm{M}} = \bm{I} - (\bm{I} + e^{-k_0\bm{\sigma}})\cdot\bm{M} + D\nabla^2\bm{M}.
\end{equation}
The over circle represents the corotational derivative $\accentset{\circ}{\bm{A}} = \partial_{t}\bm{A} + {\bf{v}}\cdot\nabla \bm{A} + \bm{\omega}\cdot\bm{A} - \bm{A}\cdot\bm{\omega}$, where $\bm{\omega} = (1/2)(\nabla {\bf{v}} - (\nabla {\bf{v}})^T)$ is the vorticity tensor.
