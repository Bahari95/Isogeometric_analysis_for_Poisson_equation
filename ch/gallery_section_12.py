__all__ = ['assemble_vector_in']

#---Assemble rhs of L2 projection
#==============================================================================
from pyccel.decorators import types

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_in(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, rhs):

    from numpy import zeros
    from numpy import random
    from numpy import sin
    from numpy import exp
    from numpy import log
    from numpy import pi
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lvalues_u  = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            f                = (2.*random.rand()-1.)*0.05 +0.63
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    x = points_1[ie1, g1]
                    y = points_2[ie2, g2]
                    lvalues_u[g1,g2] = f #exp(-20*(x-0.5-0.25*sin(2*pi*y)*sin(0.6*pi*(1.+50.*0.)))**2)
                    
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            #..
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ... rhs
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol 

                    rhs[i1+p1,i2+p2] += v   
    # ...

from simplines import pyccel_sol_field_2d
#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
import numpy                        as     np
font = {'family': 'serif', 
	 'color':  'k', 
	 'weight': 'normal', 
	 'size': 25, 
		 } 

tont = {'family': 'serif', 
	 'color':  'k', 
	 'weight': 'normal', 
	 'size': 18, 
		 } 
def plot_res(V01, V10, VPh, xu_ch, x11_mae, x12_mae, xh_n_ch, u_Pr0, nbpts, save_i, t, n_iter, levels, Sol_CH, Sol_ACH, GL_free_energy, GL_free_Adenergy, rho_h = None):    

	degree = VPh.degree[0]
	nbasis = VPh.nbasis
	n_plt  = 1
	if False :
	  #pl_rr   += 100
	  np.savetxt('data/x02_'+str(save_i)+'.txt', x02, fmt='%.2e')
	  np.savetxt('data/mapp1_'+str(save_i)+'.txt', x11_mae, fmt='%.2e')
	  np.savetxt('data/mapp2_'+str(save_i)+'.txt',  x12_mae, fmt='%.2e')
	  np.savetxt('data/sol_uni_'+str(save_i)+'.txt', xu_ch, fmt='%.2e')
	  np.savetxt('data/sol_ad_'+str(save_i)+'.txt', xh_n_ch, fmt='%.2e')
	  np.savetxt('data/density_'+str(save_i)+'.txt', rho_h, fmt='%.2e')

	#-------------------------------------------------------------------------------------------
	u_Pr,a,b,X, Y    = pyccel_sol_field_2d((nbpts,nbpts),      xu_ch, VPh.knots, VPh.degree)	
	u_ad_Pr = pyccel_sol_field_2d((nbpts,nbpts),    xh_n_ch, VPh.knots, VPh.degree)[0]
	#++++++++++++++++++++++++++++++
	#--Solution of MAE equation (optimal mapping)
	Xmae,uxx,uxy = pyccel_sol_field_2d((nbpts,nbpts),  x11_mae , V01.knots, V01.degree)[:-2]
	Ymae,uyx,uyy = pyccel_sol_field_2d((nbpts,nbpts),  x12_mae , V10.knots, V10.degree)[:-2]
	#...
	Jac_MAE = uxx*uyy- uxy*uyx
	print('>-------------------------Quality of the optimal mapping-----------------------   Jacobian min = {}  Jacobian max = {}'.format(np.min(Jac_MAE), np.max(Jac_MAE)))
	#+++-----------------------------------------------------------------------------------
	time      = np.format_float_scientific(t, unique=False, precision=3)
	# ... Statistical moment
	# -----
	fig = plt.figure() 
	plt.plot(n_iter, Sol_ACH, '*-k', linewidth = 2., label='$\mathbf{Ad}$')
	plt.plot(n_iter, Sol_CH, 'o-b', linewidth = 2., label='$\mathbf{Un}$')
	plt.xscale('log')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{time}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{M_2}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('statistical_moment_2')
	plt.show(block=False)
	plt.close()
	# -----
	fig = plt.figure() 
	plt.plot(n_iter, GL_free_Adenergy,  '--*k', label = '$\mathbf{GL-free-energy Ad}$')
	plt.plot(n_iter, GL_free_energy,  '--ob', label = '$\mathbf{GL-free-energy}$' )
	plt.xscale('log')
	plt.xlabel('$\mathbf{time}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{E}$',  fontweight ='bold', fontdict=font)
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('GL_free_energy.png')
	plt.show(block=False)
	plt.close()
	# -----
	fig =plt.figure() 
	plt.plot(Ymae[70,:], u_ad_Pr[70,:], 'o-k', linewidth = 2., label='$\mathbf{Ad}$')
	plt.plot(Y[70,:], u_Pr[70,:], '*-b', linewidth = 2., label='$\mathbf{Un}$')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{m}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{C^h}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('projection.png')
	plt.show(block=False)
	plt.close()
	# -----
	fig =plt.figure() 
	plt.plot(Ymae[10,:], u_ad_Pr[40,:], 'o-k', linewidth = 2., label='$\mathbf{Ad}$')
	plt.plot(Y[10,:], u_Pr[40,:], '*-b', linewidth = 2., label='$\mathbf{Un}$')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{m}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{C^h}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('projection10.png')
	plt.show(block=False)
	plt.close()
	# -----
	fig =plt.figure() 
	plt.plot(Ymae[110,:], u_ad_Pr[110,:], 'o-k', linewidth = 2., label='$\mathbf{Ad}$')
	plt.plot(Y[110,:], u_Pr[110,:], '*-b', linewidth = 2., label='$\mathbf{Un}$')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{m}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{C^h}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('projection110.png')
	plt.show(block=False)
	plt.close()
	# -----
	# -----
	'''
	fig =plt.figure() 
	plt.plot(Ymae[28,:], u_ad_Pr[28,:], 'o-k', linewidth = 2., label='$\mathbf{Ad-x=}0.2$')
	plt.plot(Ymae[70,:], u_ad_Pr[70,:], '*-b', linewidth = 2., label='$\mathbf{Ad-x=}0.5$')
	plt.plot(Ymae[85,:], u_ad_Pr[85,:], 'v-r', linewidth = 2., label='$\mathbf{Ad-x=}0.61$')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{m}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{C^h}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('projection_ad.png')
	plt.show(block=True)
	plt.close()
	
	fig =plt.figure() 
	plt.plot(Y[28,:], u_Pr[28,:], 'o-k', linewidth = 2., label='$\mathbf{Un-x=}0.2$')
	plt.plot(Y[70,:], u_Pr[70,:], '*-b', linewidth = 2., label='$\mathbf{Un-x=}0.5$')
	plt.plot(Y[85,:], u_Pr[85,:], 'v-r', linewidth = 2., label='$\mathbf{Un-x=}0.61$')
	plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
	plt.xlabel('$\mathbf{m}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{C^h}$',  fontweight ='bold', fontdict=font)
	plt.legend(fontsize="15")
	fig.tight_layout()
	plt.savefig('projection_un.png')
	plt.show(block=True)
	plt.close()
	# ...
	figtitle        = 'Cahn_haliard_equation '
	fig, axes       = plt.subplots( 1, 2, figsize=[12,12], num=figtitle )
	for ax in axes:
	  ax.set_aspect('equal')
	axes[0].set_title( 'With adaptive meshes at t= {}'.format(time) )
	im2 = axes[0].contourf( Xmae, Ymae, u_ad_Pr, levels, cmap= 'jet')
	divider = make_axes_locatable(axes[0]) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(im2, cax=cax)   
	x_pl = np.linspace(-0.01,1.01,100)
	axes[0].plot(x_pl*0.+0.5, x_pl, 'o-k', linewidth = 3.)
	axes[1].plot(x_pl*0.+0.5, x_pl, '*-k', linewidth = 3.)
	axes[0].text(0.44, -0.05,'$\mathbf{x}$=0.5',fontdict=tont)
	axes[1].text(0.44, -0.05,'$\mathbf{x}$=0.5',fontdict=tont)


	axes[0].plot(x_pl*0.+0.61, x_pl, 'o-k', linewidth = 3.)
	axes[1].plot(x_pl*0.+0.61, x_pl, '*-k', linewidth = 3.)
	axes[0].text(0.55, -0.05,'$\mathbf{x}$=0.61',fontdict=tont)
	axes[1].text(0.55, -0.05,'$\mathbf{x}$=0.61',fontdict=tont)

	axes[0].plot(x_pl*0.+0.2, x_pl, 'o-k', linewidth = 3.)
	axes[1].plot(x_pl*0.+0.2, x_pl, '*-k', linewidth = 3.)
	axes[0].text(0.24, -0.05,'$\mathbf{x}$=0.2',fontdict=tont)
	axes[1].text(0.24, -0.05,'$\mathbf{x}$=0.2',fontdict=tont)
				
	axes[0].axis('off')
	axes[1].axis('off')
	
	axes[1].set_title( 'With uniform mesh at t= {}'.format(time) )
	im3 = axes[1].contourf( X, Y, u_Pr, levels, cmap= 'jet')
	divider = make_axes_locatable(axes[1]) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(im3, cax=cax)

	fig.tight_layout()
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('figs/u_{}.png'.format(save_i))
	plt.show(block=True)
	#plt.pause(0.3)
	plt.close()
	'''
	# ...
	fig =plt.figure() 
	for i in range(nbpts):
	   phidx = Xmae[:,i]
	   phidy = Ymae[:,i]

	   plt.plot(phidx, phidy, '-k', linewidth = 0.25)
	for i in range(nbpts):
	   phidx = Xmae[i,:]
	   phidy = Ymae[i,:]

	   plt.plot(phidx, phidy, '-k', linewidth = 0.25)
	#axes[0].axis('off')
	plt.margins(0,0)
	fig.tight_layout()
	if save_i%n_plt == 0:
		plt.savefig('figs/meshes_{}.png'.format(save_i))
	plt.show(block=False)
	plt.close()
	# 0....
	fig, axes =plt.subplots() 
	im2 = plt.contourf( Xmae, Ymae, u_ad_Pr, levels, cmap= 'jet')
	plt.xlabel('$\mathbf{x}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{y}$',  fontweight ='bold', fontdict=font)
	divider = make_axes_locatable(axes) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(im2, cax=cax) 
	fig.tight_layout()
	if save_i%n_plt == 0:
		plt.savefig('figs/ad_sol_{}.png'.format(save_i))
	plt.show(block=False)
	plt.close()
	# ...
	fig , axes=plt.subplots() 
	im2 = plt.contourf( X, Y, u_Pr, levels, cmap= 'jet')
	plt.xlabel('$\mathbf{x}$',  fontweight ='bold', fontdict=font)
	plt.ylabel('$\mathbf{y}$',  fontweight ='bold', fontdict=font)
	divider = make_axes_locatable(axes) 
	cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
	plt.colorbar(im2, cax=cax) 
	fig.tight_layout()
	if save_i%n_plt == 0:
		plt.savefig('figs/un_sol_{}.png'.format(save_i))
	plt.show(block=False)
	plt.close()
	# ...
	if rho_h is not None:
		# ...
		Moni    = pyccel_sol_field_2d((nbpts,nbpts),      rho_h, VPh.knots, VPh.degree)[0]
		fig , axes=plt.subplots() 
		im2 = plt.contourf( X, Y, Moni, cmap= 'jet')
		plt.xlabel('$\mathbf{x}$',  fontweight ='bold', fontdict=font)
		plt.ylabel('$\mathbf{y}$',  fontweight ='bold', fontdict=font)
		divider = make_axes_locatable(axes) 
		cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
		plt.colorbar(im2, cax=cax) 
		fig.tight_layout()
		if save_i%n_plt == 0:
			plt.savefig('figs/moni_{}.png'.format(save_i))
		plt.show(block=False)
		plt.close()
	return 0
