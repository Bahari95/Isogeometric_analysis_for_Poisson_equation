from   simplines                    import compile_kernel
from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
from   simplines                    import StencilMatrix
from   simplines                    import StencilVector

# ... Using Kronecker algebra accelerated with Pyccel fast_PMAE_mesh_adaptation.py
from   kronecker.fast_diag          import Poisson

from gallery_section_06             import assemble_stiffnessmatrix1D
from gallery_section_06             import assemble_massmatrix1D
from gallery_section_06             import assemble_matrix_ex01
from gallery_section_06             import assemble_matrix_ex02
#..
from gallery_section_06             import assemble_vector_ex01
from gallery_section_06             import assemble_vector_ex00

#...
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex01, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex02, arity=1)
#..
assemble_rhs_in      = compile_kernel(assemble_vector_ex00, arity=1)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)

from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import sqrt
import numpy                        as     np
#...
import timeit
import time

#==============================================================================
#.......PMAE_SOLVER ALGORITHM
#==============================================================================
class PMAE_SOLVER(object):

    def __init__(self, V1_rho, V2_rho, degree, nelements, quad_degree):

       self.epsilon = 1.
       self.gamma   = 1.
       # create the spline space for each direction --- MAE equation
       V1  = SplineSpace(degree= degree,   nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V2  = SplineSpace(degree= degree,   nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V3  = SplineSpace(degree= degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V4  = SplineSpace(degree= degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)

       # create the tensor space
       V00 = TensorSpace(V1, V2)
       V11 = TensorSpace(V3, V4)
       V01 = TensorSpace(V1, V3)
       V10 = TensorSpace(V4, V2)
       
       V   = TensorSpace(V3, V4, V1_rho, V2_rho)
       #----------------------------------------------------------------------------------------------
       # ... Strong form of Neumann boundary condition which is Dirichlet because of Mixed formulation
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #..
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)

       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #..
       u_01.from_array(V01, x_D)
       u_10.from_array(V10, y_D)

       #... We delete the first and the last spline function
       #.. as a technic for applying Neumann boundary condition
       #.in a mixed formulation

       #..Stiffness and Mass matrix in 1D in the first deriction
       D1         = assemble_mass1D(V3)
       D1         = D1.tosparse()
       D1         = D1.toarray()
       D1         = csr_matrix(D1)
       #___
       M1         = assemble_mass1D(V1)
       M1         = M1.tosparse()
       m1         = M1
       M1         = M1.toarray()[1:-1,1:-1]
       M1         = csc_matrix(M1)
       m1         = csr_matrix(m1)

       #..Stiffness and Mass matrix in 1D in the second deriction
       D2         = assemble_mass1D(V4)
       D2         = D2.tosparse()
       D2         = D2.toarray()
       D2         = csr_matrix(D2)
       #___
       M2         = assemble_mass1D(V2)
       M2         = M2.tosparse()
       m2         = M2
       M2         = M2.toarray()[1:-1,1:-1]
       M2         = csc_matrix(M2)
       m2         = csr_matrix(m2)

       #...
       R1         = assemble_matrix_ex01(V01)
       R1         = R1.toarray()
       R1         = R1.reshape(V01.nbasis)
       R1         = R1[1:-1,:]
       R1         = csr_matrix(R1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)

       #...step 0.1
       mats_1     = [M1, 0.5*M1]
       mats_2     = [D2, 0.5*D2]

       # ...Fast Solver
       self.poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, 0.5*D1]
       mats_2     = [M2, 0.5*M2]

       # ...Fast Solver
       self.poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 2
       K1         = assemble_stiffness1D(V4)
       K1         = K1.tosparse()
       # ...
       K2         = assemble_stiffness1D(V3)
       K2         = K2.tosparse()
       mats_1     = [D1, self.epsilon*self.gamma*K1]
       mats_2     = [D2, self.epsilon*self.gamma*K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2, tau = self.epsilon)
       #...
       self.b11   = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       self.b12   = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())

       #___
       self.x11_1  = kron(R1, D2)
       self.x12_1  = kron(D1, R2.T)
       #___
       # ... for assembling residual
       self.M_res  = kron(M1, D2)
       #___       
       self.spaces = [V1, V2, V3, V4, V11, V01, V10, V]
       self.poisson= poisson
       
    def solve(self, u_H =  None, x2 = None, niter  = None, tol = None):
       
       #...
       if tol is None :
          tol     = 1e-5
       dt         = 0.5
       epsilon    = self.epsilon
       gamma      = self.gamma
       #...
       V1, V2, V3, V4, V11, V01, V10, V  = self.spaces[:]
       poisson       = self.poisson
       M_res         = self.M_res
       
       # ... for Two or Multi grids
       if x2 is None :    
           niter         = 10
           u             = StencilVector(V11.vector_space)
           # ...
           x_2           = zeros((V1.nbasis-2)*V3.nbasis)
           
           #---Assembles a right hand side of Poisson equation
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs_in(V, fields = [u_H], value = [epsilon, gamma, dt], out= rhs )
           b            = rhs.toarray()
           #___
           # ... Solve first system
           x2           = poisson.solve(b)
           #___
           u.from_array(V11, x2.reshape(V11.nbasis))

       else           :       
           u            = StencilVector(V11.vector_space)
           # ...
           u.from_array(V11, x2.reshape(V11.nbasis))
           # ...
           x_2          = zeros((V1.nbasis-2)*V3.nbasis)
           #..Residual 
           #var_1        = self.poisson_c1.solve(self.b11 - self.x11_1.dot(x2))
           #dx           = var_1-x_2
           #x_2[:]       = var_1[:]
           #i            = 0
           #niter        = 0
           #l2_residual  = 0.
       for i in range(niter):
           #---Assembles a right hand side of Poisson equation
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs(V, fields = [u, u_H], knots = True, value = [epsilon, gamma, dt], out= rhs )
           b            = rhs.toarray()
           # ... Solve first system
           x2           = poisson.solve(b)
           #___
           u.from_array(V11, x2.reshape(V11.nbasis))
           #..Residual 
           var_1        = self.poisson_c1.solve(self.b11 - self.x11_1.dot(x2))
           dx           = var_1[:]-x_2[:]
           x_2[:]       = var_1[:]
           #... Compute residual for L2
           l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
           if l2_residual < tol:
              break
       #... last step
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #___
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)
       #___
       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #___
       x_D[1:-1,:]  =  self.poisson_c1.solve(self.b11 - self.x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])
       u_01.from_array(V01, x_D)
       #___
       y_D[:,1:-1]  =  self.poisson_c2.solve(self.b12 - self.x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
       u_10.from_array(V10, y_D)
       print('-----> N-iter in PMAE_SOLVER ={} -----> l2_residual= {}'.format(i, l2_residual))
       return u_01, u_10, x_D, y_D, x2

