# Missing SPH Equations for Documentation

Based on your current markdown file, I've identified several key equations missing from your SPH documentation. Here they are with LaTeX formatting:

## 1. Density Estimation Equation

**Missing**: The fundamental SPH density estimation formula

$$\rho_i = \sum_j m_j W(|\vec{r}_i - \vec{r}_j|, h)$$

**Where to integrate**: This should go right after you mention "Thus, the sum is weighted by a function of the particle separation, known as the kernel function, $W$." This equation shows exactly HOW the density is calculated using the kernel function.

**Setup needed**: 
- $\rho_i$ = density at particle $i$
- $m_j$ = mass of neighboring particle $j$ 
- $|\vec{r}_i - \vec{r}_j|$ = distance between particles $i$ and $j$
- The sum is over all particles within the kernel support ($2h$)

## 2. Pressure Equation

**Missing**: How pressure relates to density (equation of state)

From your code, you're using a simplified form of the Tait equation:

$$P_i = \frac{\rho_i c_s^2}{\gamma}$$

**Alternative form** (commented in your code but more standard):
$$P_i = \frac{\rho_0 c_s^2}{\gamma} \left[\left(\frac{\rho_i}{\rho_0}\right)^\gamma - 1\right]$$

**Where to integrate**: Right before or after the SPH force equation, since pressure $P_i$ appears in the force calculation.

**Setup needed**:
- $P_i$ = pressure at particle $i$
- $\rho_i$ = density at particle $i$
- $\rho_0$ = reference density
- $c_s$ = sound speed
- $\gamma$ = stiffness parameter (typically 5/3 for adiabatic gas, 7 for water)

## 3. Kernel Gradient Equation

**Missing**: How to calculate $\nabla_i W_{ij}$ that appears in the force equation

$$\nabla_i W_{ij} = \frac{dW}{dr}\bigg|_{r=|\vec{r}_{ij}|} \hat{\vec{r}}_{ij}$$,

which basically says that you take the derivative of $W$ with respect to $r$, which is spherically symmetric. Then you multiply it by the unit vector along the direction separating particles $i$ and $j$ to give the direction of the gradient. The deri

where:
$$\frac{dW}{dr} = \frac{\sigma}{h^{d+1}} \begin{cases}
-3q + \frac{9}{4}q^2 & \text{if } 0 \leq q \leq 1 \\
-\frac{3}{4}(2 - q)^2 & \text{if } 1 < q \leq 2 \\
0 & \text{if } q > 2
\end{cases}$$

**Where to integrate**: After the kernel function definition, since the gradient is needed for the force calculation.

**Setup needed**:
- $\vec{r}_{ij} = \vec{r}_i - \vec{r}_j$ = separation vector
- $\hat{\vec{r}}_{ij} = \vec{r}_{ij}/|\vec{r}_{ij}|$ = unit separation vector
- $dW/dr$ = radial derivative of kernel function

## 4. Complete SPH Force Derivation (Optional Enhancement)

**Missing**: The connection between continuous and discrete forms

The continuous momentum equation:
$$\frac{D\vec{v}}{Dt} = -\frac{1}{\rho}\nabla P$$

becomes in SPH:
$$\frac{d\vec{v}_i}{dt} = -\sum_j m_j \left( \frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} \right) \nabla_i W_{ij}$$

**Where to integrate**: This could provide nice context between the continuous fluid equations and your discrete SPH implementation.

## Integration Suggestions

1. **Start with density equation** right after introducing the kernel concept
2. **Add pressure equation** to show how forces depend on density  
3. **Include kernel gradient** as a technical detail after the kernel function
4. **Link everything together** by showing how density → pressure → force

This creates a logical flow: "We need density to calculate forces → Here's how we estimate density → Here's how density gives pressure → Here's how pressure creates forces."

The widget perfectly demonstrates the first step (density estimation with different $h$ values), making it a great visual complement to these equations!