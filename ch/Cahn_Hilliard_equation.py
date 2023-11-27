from simplines import compile_kernel, apply_periodic

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import least_square_Bspline

#.. Prologation by knots insertion matrix
from   simplines                   import prolongation_matrix

#====================================================
#.... Cahn_hilliard utilities
from gallery_section_08 import assemble_matrix_ex01 #---1 : In uniform mesh 
from gallery_section_08 import assemble_vector_ex01 #---1 : In uniform mesh
from gallery_section_08 import assemble_norm_ex01   #---1 : In uniform mesh 

assemble_stiffness = compile_kernel(assemble_matrix_ex01, arity=2)
assemble1_rhs      = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2   = compile_kernel(assemble_norm_ex01, arity=1)
#...
from gallery_section_08 import assemble_matrix_ex04 #---2 : In adapted mesh
from gallery_section_08 import assemble_vector_ex02 #---2 : In adapted mesh
from gallery_section_08 import assemble_norm_ex02   #---2 : In adapted mesh

assemble2_rhs       = compile_kernel(assemble_vector_ex02, arity=1)
assemble2_stiffness = compile_kernel(assemble_matrix_ex04, arity=2)
assemble2_norm_l2   = compile_kernel(assemble_norm_ex02, arity=1)

#................................................
# ... Import mesh adaptation solver and monitor function 
#from  MFMAE_mesh_adaptation_solver  import MFMAE_SOLVER
from  PMAE_mesh_adaptation_solver   import PMAE_SOLVER
from  MFMAE_mesh_adaptation_solver  import density_from_solution
#................................................
from  gallery_section_12            import plot_res
from  gallery_section_12            import assemble_vector_in

assemble_rhs_in      = compile_kernel(assemble_vector_in, arity=1)

#=================================
from scipy.sparse        import kron
from scipy.sparse        import csr_matrix
from scipy.sparse        import csc_matrix, linalg as sla
from kronecker.kronecker import vec_2d
from kronecker.fast_diag import Poisson
from numpy               import zeros, linalg, asarray, sqrt
#+-
import timeit
import time
import numpy as np

#==============================================================================
#.......Pde ALGORITHM
#==============================================================================
class Cahn_hilliard_equation(object):

	def __init__(self, degree, nelements, quad_degree, periodic, dt, Pi, alpha, theta, N_iter  = None):
	
		# create the spline space for each direction
		V1             = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=periodic[0], quad_degree = quad_degree)
		V2             = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=periodic[1], quad_degree = quad_degree)
		Vh             = TensorSpace(V1, V2)
		if N_iter is None :
			N_iter = 100	
		tol            = 1e-6 
		# -------------------------------------
		self.spaces    = [V1, V2, Vh]
		self.V_ad      = TensorSpace(Pi.spaces[0], Pi.spaces[1], Pi.spaces[2], Pi.spaces[3], V1, V2)
		self.V01       = Pi.spaces[5]
		self.V10       = Pi.spaces[6]
		self.nbasis    = [V1.nbasis-V1.degree, V2.nbasis-V2.degree]
		self.periodic  = periodic
		self.N_iter    = N_iter
		self.tol       = tol
		self.alpha     = alpha 
		self.theta     = theta
		self.dt        = dt

	def solve_uniform_mesh(self, u, xh):

		#... two-stage pre-dictor–multicorrector algorithm
		V1, V2, V= self.spaces[:]
		periodic = self.periodic
		alpha    = self.alpha 
		theta    = self.theta
		dt       = self.dt
		N_iter   = self.N_iter
		tol      = self.tol
		nbasis   = self.nbasis

		u_f        = StencilVector(V.vector_space)

		#... Step 1 : Initialization
		xu_n       = zeros(V.nbasis)
		xu_n[:,:]  = xh[:,:]

		#___
		u_f.from_array(V, xu_n )
		#... step 2 : Multicoerrector :
		for i in range(0, N_iter): 
		     
			#___ step (b): implicite time schem level

			stiffness  = assemble_stiffness(V, fields=[u_f], value = [dt, alpha, theta])
			M          = apply_periodic(V, stiffness, periodic)
			#...
			rhs        = assemble1_rhs( V, fields=[u_f, u], value = [dt, alpha, theta])
			rhs        = apply_periodic(V, rhs, periodic)

			#--Solve a linear system
			b          = -1.*rhs
			#++ 
			lu         = sla.splu(csc_matrix(M))
			d_thx      = lu.solve(b)
			#print('CPU-time  SUP_LU== ', time.time()- start)
			d_tx       = d_thx.reshape(nbasis)                    
			d_tx       = apply_periodic(V, d_tx, periodic, update= True)
			#___ step (c): update
			xu_n[:,:] = xu_n[:,:] + d_tx[:,:]
			Res        = np.max(np.absolute(d_thx))
			#___ step (a)
			u_f.from_array(V, xu_n )

			if Res < tol or Res > 1e10:
				break
		print('perform the iteration number : = {} Residual  = {}'.format( i, Res))
		# ...
		u.from_array(V, xu_n)
		# ...
		Norm      = assemble_norm_l2(V, fields=[u], value = [alpha, theta])
		norm      = Norm.toarray()[1]
		return u, xu_n, Res, norm

	def solve_adapted_mesh(self, u11_mae, u12_mae, u, u01_mae, u10_mae):

		#... two-stage pre-dictor–multicorrector algorithm
		V1, V2, V= self.spaces[:] 
		Vh       = self.V_ad
		periodic = self.periodic
		alpha    = self.alpha 
		theta    = self.theta
		dt       = self.dt
		N_iter   = self.N_iter
		tol      = self.tol
		nbasis   = self.nbasis
		
		u_f        = StencilVector(V.vector_space)
		# ...
		xh = u.toarray().reshape(V.nbasis)         
		# ... Step 1 : Initialization
		xu_n       = zeros(V.nbasis)
		xu_n[:,:]  = xh[:,:]
		# ... :  u_f in uniform mesh
		u_f.from_array(V, xu_n ) 
		for i in range(0, N_iter): 

			#___ step (b): Implicite time schem level  
			start = time.time()         
			stiffness  = StencilMatrix(V.vector_space, V.vector_space)
			stiffness  = assemble2_stiffness(Vh, fields=[u11_mae, u12_mae, u_f, u01_mae, u10_mae], value = [dt, alpha, theta], out = stiffness)
			M          = apply_periodic(V, stiffness, periodic)
			#...
			rhs        = StencilVector(V.vector_space)          
			rhs        = assemble2_rhs( Vh, fields=[u11_mae, u12_mae, u_f, u, u01_mae, u10_mae], value = [dt, alpha, theta], out = rhs)
			rhs        = apply_periodic(V, rhs, periodic)

			#--Solve a linear system
			b          = -1.*(rhs)
			# ...
			M          = csc_matrix(M)
			# +++ 
			#start = time.time()         
			lu         = sla.splu(M)
			d_thx      = lu.solve(b)
			#print('\n time consumed to solve linear system ==================================', time.time() - start,'\n')
			# ...
			d_tx       = d_thx.reshape(nbasis)                    
			d_tx       = apply_periodic(V, d_tx, periodic, update= True)
			#___ step (c): update
			xu_n[:,:] = xu_n[:,:] + d_tx[:,:]
			Res        = np.max(np.absolute(d_thx))
			#___ step (a)
			u_f.from_array(V, xu_n )

			if Res < tol or Res > 1e10:
				break

		print('perform the iteration number : = {} Residual  = {}'.format( i, Res))
		# ...
		u.from_array(V, xu_n)
		# ...
		Norm       = StencilVector(V.vector_space)
		Norm       = assemble2_norm_l2(Vh, fields=[u11_mae, u12_mae, u], value = [alpha, theta], out = Norm)
		norm       = Norm.toarray()[1]
		return u, xu_n, Res, norm

# ... EDPs data
alpha           = 3000.
theta           = 3./2 

# ... Galerkin initialization
mdegree         = 4
degree          = 2
quad_degree     = max(degree, mdegree)+3
nelements       = 32
periodic        = [True, True]

# ...
dt              = 1e-8
t               = 0.
levels          = list(np.linspace(-0.0,1.0,100))
# ...
nbpts           = 140    # for plot
t_max           = 1.      # for time
Sol_CH          = []
Sol_ACH         = []
n_iter          = []
GL_free_energy  = []
GL_free_Adenergy= []

#=======================================================
# ---- Initialization
#=======================================================
#--- Computes the initial CH randem solution and B-spline solver for densioty function
Pr     = density_from_solution(degree, nelements, quad_degree, periodic)

#---  MIXED FORMUATION MAE solver
#Pi     = MFMAE_SOLVER( Pr.spaces[0], Pr.spaces[1], mdegree, nelements, quad_degree)

#--- Parabolic MAE solver
Pi     = PMAE_SOLVER( Pr.spaces[0], Pr.spaces[1], mdegree, nelements, quad_degree)

#--- Cahn_hiliard solver
Pde    = Cahn_hilliard_equation(degree, nelements, quad_degree, periodic, dt, Pi, alpha, theta)

dtu01_mae           = StencilVector(Pde.V01.vector_space)
dtu10_mae           = StencilVector(Pde.V10.vector_space)

if True :
	#=======================================================
	# --- initial Guess
	#=======================================================

	u_ch, xu_ch, M_ms = Pr.initial_guess()

	#... compute the initial GL energie
	norm      = assemble_norm_l2(Pr.spaces[2], fields=[u_ch], value = [alpha, theta])
	norm      = norm.toarray()[1]

	#... Fixed for computing statistical moment
	u_Pr0      = xu_ch
	#---Adaptive meshes
	u_n_ch     = u_ch

	#-------------------------------------------------------
	#..... computing the initial uniform mapping
	u11_mae, u12_mae, x11_mae, x12_mae, x02 = Pi.solve(u_H = u_ch)

	# ...
	du_ch     = (u_Pr0[:-degree,:-degree]-xu_ch[:-degree,:-degree]).reshape(Pde.nbasis[0]*Pde.nbasis[1])
	Sol_CH.append((M_ms.dot(du_ch)).dot(du_ch) )
	Sol_ACH.append((M_ms.dot(du_ch)).dot(du_ch))
	GL_free_energy.append(norm)     
	GL_free_Adenergy.append(norm)
	n_iter.append(np.exp(float(np.format_float_scientific( t, unique=False, precision=2)))-1.)   
	plot_res(Pde.V01, Pde.V10, Pde.spaces[2], xu_ch, x11_mae, x12_mae, xu_ch, u_Pr0, nbpts, 0, t, n_iter, levels, Sol_CH, Sol_ACH, GL_free_energy, GL_free_Adenergy)
		  	  	  
	pl_r   = 1
	save_i = 1
	ii     = 0

if False :
	M_ms = Pr.initial_guess()[2]
	pl_r    = 43989
	save_i  = 5170
	ii      = 43989
	t       = 0.010819699999998692
	'''
	pl_r    = 38388
	save_i  = 5114
	ii      = 38388
	t       = 0.0052177000000005564
	'''
	x02     = np.loadtxt('data/x02_'+str(save_i)+'.txt')
	x11_mae = np.loadtxt('data/mapp1_'+str(save_i)+'.txt')
	x12_mae = np.loadtxt('data/mapp2_'+str(save_i)+'.txt')
	xu_ch   = np.loadtxt('data/sol_uni_'+str(save_i)+'.txt')
	xh_n_ch = np.loadtxt('data/sol_ad_'+str(save_i)+'.txt')
	u_Pr0   = np.loadtxt('data/u_Pr0_'+str(save_i)+'.txt')
	Sol_CH  = list(np.loadtxt('data/Sol_CH_'+str(save_i)+'.txt')[0:save_i])
	Sol_ACH = list(np.loadtxt('data/Sol_ACH_'+str(save_i)+'.txt')[0:save_i])
	GL_free_energy = list(np.loadtxt('data/GL_free_energy_'+str(save_i)+'.txt')[0:save_i])
	GL_free_Adenergy = list(np.loadtxt('data/GL_free_Adenergy_'+str(save_i)+'.txt')[0:save_i])
	n_iter  = list(np.loadtxt('data/n_iter_'+str(save_i)+'.txt')[0:save_i])
	u_ch    = StencilVector(Pde.spaces[2].vector_space)
	u_ch.from_array(Pde.spaces[2], xu_ch)
	
	u11_mae = StencilVector(Pde.V01.vector_space)
	u11_mae.from_array(Pde.V01, x11_mae)
	u12_mae = StencilVector(Pde.V10.vector_space)
	u12_mae.from_array(Pde.V10, x12_mae)
	
	u_n_ch  = StencilVector(Pde.spaces[2].vector_space)
	u_n_ch.from_array(Pde.spaces[2], xh_n_ch)
	print(t, '  ', pl_r, '  ', save_i, '  ', ii, '\n')
	save_i += 1
	pl_r   += 1
	x02     =  None
	#plot_res(Pde.V01, Pde.V10, Pde.spaces[2], xu_ch, x11_mae, x12_mae, xh_n_ch, u_Pr0, nbpts, save_i, t, n_iter, levels, Sol_CH, Sol_ACH, GL_free_energy, GL_free_Adenergy, rho_h = rho_h)

while t <= t_max:
	t           += dt
	dtu01_mae.from_array(Pde.V01, x11_mae)
	dtu10_mae.from_array(Pde.V10, x12_mae)
	print('\n\n', '================= In Uniform mesh  ============================')
	#-------------------------------------------
	u_ch, xu_ch, Res1, norm_f = Pde.solve_uniform_mesh(u_ch, xu_ch)

	# ... In adaptive meshes
	print('================= In adaptive meshes ===========================')
	#-----------------------------------------------------------
	# ... computation of the optimal mapping using last solution 
	rho_st, rho_h                           = Pr.density(Pde.V_ad, u11_mae, u12_mae, u_n_ch)
	#plot_res(Pde.V01, Pde.V10, Pde.spaces[2], xu_ch, x11_mae, x12_mae, xh_n_ch, u_Pr0, nbpts, save_i, t, n_iter, levels, Sol_CH, Sol_ACH, GL_free_energy, GL_free_Adenergy, rho_h = rho_h)
	# ... Computes the optimal mapping using initial solution 
	u11_mae, u12_mae, x11_mae, x12_mae, x02 = Pi.solve(u_H = rho_st, x2 = x02, niter  = 2)

	# ...   solve Cahn Hilliard in adapted meshes
	u_n_ch, xh_n_ch, Res, norm_ad           = Pde.solve_adapted_mesh(u11_mae, u12_mae, u_n_ch, dtu01_mae, dtu10_mae)
	#------------
	if  Res + Res1 > 1. :
		print("Sorry. Your settings or the regularity assumption are not working !!!")
		break
	else :
	  np.savetxt('data/x02_'+str(save_i)+'.txt', x02, fmt='%.2e')
	  np.savetxt('data/mapp1_'+str(save_i)+'.txt', x11_mae, fmt='%.2e')
	  np.savetxt('data/mapp2_'+str(save_i)+'.txt',  x12_mae, fmt='%.2e')
	  np.savetxt('data/sol_uni_'+str(save_i)+'.txt', xu_ch, fmt='%.2e')
	  np.savetxt('data/sol_ad_'+str(save_i)+'.txt', xh_n_ch, fmt='%.2e')
	  np.savetxt('data/u_Pr0_'+str(save_i)+'.txt', u_Pr0, fmt='%.2e')
	  np.savetxt('data/Sol_CH_'+str(save_i)+'.txt', Sol_CH, fmt='%.2e')
	  np.savetxt('data/Sol_ACH_'+str(save_i)+'.txt', Sol_ACH, fmt='%.2e')
	  np.savetxt('data/GL_free_energy_'+str(save_i)+'.txt', GL_free_energy, fmt='%.2e')
	  np.savetxt('data/GL_free_Adenergy_'+str(save_i)+'.txt', GL_free_Adenergy, fmt='%.2e')
	  np.savetxt('data/n_iter_'+str(save_i)+'.txt', n_iter, fmt='%.2e')
	  np.savetxt('data/rho_h_'+str(save_i)+'.txt', n_iter, fmt='%.2e')
	  print(t, '  ', pl_r, '  ', save_i, '  ', ii, '\n')
	if ii == pl_r :
		pl_r   += 10
		#+++++++++++++++++++++++++++++
		du_ch     = (u_Pr0[:-degree,:-degree]-xu_ch[:-degree,:-degree]).reshape(Pde.nbasis[0]*Pde.nbasis[1])
		Sol_CH.append((M_ms.dot(du_ch)).dot(du_ch) )
		du_ch     = (u_Pr0[:-degree,:-degree]-xh_n_ch[:-degree,:-degree]).reshape(Pde.nbasis[0]*Pde.nbasis[1])
		Sol_ACH.append((M_ms.dot(du_ch)).dot(du_ch) )
		GL_free_energy.append(norm_f)     
		GL_free_Adenergy.append(norm_ad)
		n_iter.append(np.exp(float(np.format_float_scientific( t, unique=False, precision=2)))-1.)   
		# ...
		plot_res(Pde.V01, Pde.V10, Pde.spaces[2], xu_ch, x11_mae, x12_mae, xh_n_ch, u_Pr0, nbpts, save_i, t, n_iter, levels, Sol_CH, Sol_ACH, GL_free_energy, GL_free_Adenergy, rho_h = rho_h)
		# ...
		save_i += 1
		#_____________________________
	ii += 1
	
	
if True :   
 import imageio 
 with imageio.get_writer('u_ad.gif', mode='I') as writer: 
     for filename in ['Cahn_Hilliard/uad_{}.png'.format(i) for i in range(1,250,10)]: 
         image = imageio.imread(filename) 
         writer.append_data(image) 
     for filename in ['Cahn_Hilliard/uad_{}.png'.format(i) for i in range(250,0,3)]: 
         image = imageio.imread(filename) 
         writer.append_data(image) 
