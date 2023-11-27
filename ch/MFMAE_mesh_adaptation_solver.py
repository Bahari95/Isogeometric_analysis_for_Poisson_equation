from   simplines                    import compile_kernel
from   simplines                    import apply_periodic
from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
from   simplines                    import StencilMatrix
from   simplines                    import StencilVector

# ... Using Kronecker algebra accelerated with Pyccel fast_PMAE_mesh_adaptation.py
from   kronecker.fast_diag          import Poisson

#-----------------------------------------------------------MAE utilities
from gallery_section_10 import assemble_stiffnessmatrix1D
from gallery_section_10 import assemble_massmatrix1D
from gallery_section_10 import assemble_matrix_ex11
from gallery_section_10 import assemble_matrix_ex12
from gallery_section_10 import assemble_vector_ex0mae # .. identity mapping

assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex11, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex12, arity=1)
assemble_rhsmae_t0    = compile_kernel(assemble_vector_ex0mae, arity=1)


#++++ MAE and Density function
from gallery_section_11 import assemble_vector_ex01mae
from gallery_section_11 import assemble_vector_rhsmae

assemble_rhsmae       = compile_kernel(assemble_vector_ex01mae, arity=1)
assemble_monmae       = compile_kernel(assemble_vector_rhsmae, arity=1)

# ...
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
class MFMAE_SOLVER(object):

    # ... intensity of mesh refinement
    intensity = 50.
    def __init__(self, V_m1, V_m2, degree, nelements, quad_degree):

       #----------------------
       # create the spline space for each direction
       V1    = SplineSpace(degree= degree  , nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V2    = SplineSpace(degree= degree  , nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V3    = SplineSpace(degree= degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
       V4    = SplineSpace(degree= degree-1, nelements= nelements, nderiv = 2, quad_degree = quad_degree)

       # create the tensor space
       V00   = TensorSpace(V1, V2)
       V11   = TensorSpace(V3, V4)
       V01   = TensorSpace(V1, V3)
       V10   = TensorSpace(V4, V2)
       
       #___
       I1         = np.eye(V3.nbasis)
       I2         = np.eye(V4.nbasis)

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
       r1         = R1.T
       R1         = R1[1:-1,:].T
       R1         = csr_matrix(R1)
       r1         = csr_matrix(r1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       r2         = R2
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)
       r2         = csr_matrix(r2)

       #...step 0.1
       mats_1     = [M1, M1]
       mats_2     = [D2, D2]

       # ...Fast Solver
       poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, D1]
       mats_2     = [M2, M2]

       # ...Fast Solver
       poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 1
       M1         = sla.inv(M1)
       A1         = M1.dot(R1.T)
       K1         = R1.dot( A1)
       K1         = csr_matrix(K1)
       #___
       M2         = sla.inv(M2)
       A2         = M2.dot( R2.T)
       K2         = R2.dot( A2)
       K2         = csr_matrix(K2)

       #...step 2
       mats_1     = [D1, K1]
       mats_2     = [D2, K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2)

       #  ... Strong form of Neumann boundary condition which is Dirichlet because of Mixed formulation
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #..
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)

       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #..
       #..
       u_01.from_array(V01, x_D)
       u_10.from_array(V10, y_D)
       #...non homogenoeus Neumann boundary 
       b01        = -kron(r1, D2).dot(u_01.toarray())
       #__
       b10        = -kron(D1, r2).dot( u_10.toarray())
       b_0        = b01 + b10
       #...
       b11        = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       b12        = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())
       
       #___Solve first system
       self.r_0     =  kron(A1.T, I2).dot(b11) + kron(I1, A2.T).dot(b12) - b_0

       #___
       self.x11_1   = kron(A1, I2)
       self.x12_1   = kron(I1, A2)
       #___
       self.C1      = poisson_c1.solve(2.*b11)
       self.C2      = poisson_c2.solve(2.*b12)
       # ----
       self.V       = TensorSpace(V1, V2, V3, V4, V_m1, V_m2)
       self.spaces  = [V1, V2, V3, V4, V11, V01, V10]
       self.poisson = poisson
       self.M_res   = kron(D1, D2)
       
    def solve(self, x_2 = None, u_H = None):

        V1, V2, V3, V4, V11, V01, V10    = self.spaces[:]
        V      = self.V
        # ... for residual
        M_res  = self.M_res     
        #----------------------------------------------------------------------------------------------
        tol    = 1.e-7
        niter  = 100
        
        if x_2 is None :
            # ... for Two or Multi grids
            u11     = StencilVector(V01.vector_space)
            u12     = StencilVector(V10.vector_space)
            x11     = np.zeros(V01.nbasis) # dx/ appr.solution
            x12     = np.zeros(V10.nbasis) # dy/ appr.solution
            # ...
            u11.from_array(V01, x11)
            u12.from_array(V10, x12)
            # ...Assembles Neumann boundary conditions
            x11[-1,:]  = 1.
            x12[:,-1]  = 1.
            # .../
            x_2     = zeros(V3.nbasis*V4.nbasis)
        else:
            u11          = StencilVector(V01.vector_space)
            u12          = StencilVector(V10.vector_space)
            x11          = np.zeros(V01.nbasis) # dx/ appr.solution
            x12          = np.zeros(V10.nbasis) # dy/ appr.solution
            # ...Assembles Neumann (Dirichlet) boundary conditions
            x11[-1,:]    = 1.
            x12[:,-1]    = 1.
            # ...
            x11[1:-1,:]  =  (self.C1 - self.x11_1.dot(x_2)).reshape([V1.nbasis-2,V3.nbasis])
            u11.from_array(V01, x11)
            #___
            x12[:,1:-1]  =  (self.C2 - self.x12_1.dot(x_2)).reshape([V4.nbasis,V2.nbasis-2])
            u12.from_array(V10, x12)        

        if u_H is None:

             for i in range(niter):
           
                   #---Assembles a right hand side of Poisson equation
                   rhs          = StencilVector(V11.vector_space)
                   rhs          = assemble_rhsmae_t0(V, fields = [u11, u12], out= rhs)
                   b            = rhs.toarray()
                   #___
                   r            =  self.r_0 - b
           
                   # ... Solve first system
                   x2           = self.poisson.solve(r)
                   x2           = x2 -sum(x2)/len(x2)
                   #___
                   x11[1:-1,:]  =  (self.C1 - self.x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])      
                   u11.from_array(V01, x11)
                   #___
                   x12[:,1:-1]  =  (self.C2 - self.x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
                   u12.from_array(V10, x12)
  
                   #..Residual   
                   dx           = x2[:]-x_2[:]
                   x_2[:]       = x2[:]
           
                   #... Compute residual for L2
                   l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
            
                   if l2_residual < tol:
                       break
        else  :           
             for i in range(niter):
           
                   #---Assembles a right hand side of Poisson equation
                   rhs          = StencilVector(V11.vector_space)
                   rhs          = assemble_rhsmae(V, fields = [u11, u12, u_H], knots = True, value = [MFMAE_SOLVER.intensity], out= rhs)
                   b            = rhs.toarray()
                   #___
                   r            =  self.r_0 - b
           
                   # ... Solve first system
                   x2           = self.poisson.solve(r)
                   x2           = x2 -sum(x2)/len(x2)
                   #___
                   x11[1:-1,:]  =  (self.C1 - self.x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])      
                   u11.from_array(V01, x11)
                   #___
                   x12[:,1:-1]  =  (self.C2 - self.x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
                   u12.from_array(V10, x12)
  
                   #..Residual   
                   dx           = x2[:]-x_2[:]
                   x_2[:]       = x2[:]
           
                   #... Compute residual for L2
                   l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
            
                   if l2_residual < tol:
                       break
        print('-----> N-iter in MFMAE_SOLVER ={} -----> l2_residual= {}'.format(i, l2_residual))
        return u11, u12, x11, x12, x2
        
#================================================================================
#.. Computes the L2 projection of a solution at last iteration            
#==============================================================================
class density_from_solution(object):

	# ... left_v and right_v is the bounds of a density function [left_v, right_v]
	left_v   = 1.
	right_v  = 8.
	assert(right_v - left_v > 0.)

	def __init__(self, degree, nelements, quad_degree, periodic):
		#___
		V1             = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=periodic[0], quad_degree = quad_degree)
		V2             = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=periodic[1], quad_degree = quad_degree)
		# ...
		M1               = assemble_mass1D(V1)
		M1               = apply_periodic(V1, M1)
		M1               = csr_matrix(M1)

		M2               = assemble_mass1D(V2)
		M2               = apply_periodic(V1, M2)
		M2               = csr_matrix(M2)

		mats_1           = [0.5*M1, 0.5*M1]
		mats_2           = [M2, M2]
		# ...
		mats_1           = [0.5*M1, 0.5*M1]
		mats_2           = [M2, M2]
		# ...
		self.poisson     = Poisson(mats_1, mats_2)   
		# ...
		V                = TensorSpace(V1, V2)
		self.spaces      = [V1, V2, V] 
		self.periodic    = periodic
		self.nbasis      = [V1.nbasis-V1.degree, V2.nbasis-V2.degree]
		self.M1          = M1
		self.M2          = M2
           
	def density(self, Vh, u11_mae, u12_mae, u):

		nbasis     = (self.nbasis[0],self.nbasis[0])
		V          = self.spaces[2]
		#print('-----L2 projection of the solution in the B-spline space--------------')
		u_L2       = StencilVector(V.vector_space)
		#...
		rhs        = StencilVector(V.vector_space)
		rhs        = assemble_monmae(Vh, fields = [u11_mae, u12_mae, u], knots = True, value = [density_from_solution.left_v, density_from_solution.right_v], out= rhs)
		b          = apply_periodic( V, rhs, self.periodic)

		#---Solve a linear system
		x          = self.poisson.solve(b)
		x          = x.reshape(nbasis)
		#x          = x*13./np.max(x) + 1.- 13./np.max(x)
		x          = apply_periodic(V, x, self.periodic, update = True)
		#...
		u_L2.from_array(V, x)
		return u_L2, x

	def initial_guess(self):
		V1, V2,  V = self.spaces[:]
		M_mass     = kron(self.M1, self.M2)
		nbasis     = (self.nbasis[0],self.nbasis[0])       
		#... We delete the first and the last spline function
		#. as a technic for applying Dirichlet boundary condition
		u                = StencilVector(V.vector_space)

		#---- assemble random control Points
		#rhs              = assemble_rhs_in( V)
		#b                = apply_periodic(V, rhs, self.periodic)
		#xh               = self.poisson.solve(b)
		#xh               = xh.reshape(nbasis)
		#... this is for random coeffs
		xh               = (1.-2.*np.random.rand(nbasis[0], nbasis[1]))*0.05 + 0.63
		xh               = apply_periodic(V, xh, self.periodic, update = True)
		u.from_array(V, xh)

		return u, xh, M_mass
		
# ...
