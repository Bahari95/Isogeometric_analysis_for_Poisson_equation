__all__ = ['assemble_matrix_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex01',
           'assemble_vector_ex02',
           'assemble_norm_ex01',
           'assemble_norm_ex02'
]

#==============================================================================
from pyccel.decorators import types

#==============================================================================
#---2 : In adapted mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:,:,:]')
def assemble_matrix_ex01(ne1, ne2, p1, p2,  spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, vector_u, dt, alpha, theta, matrix):

    # ... sizes
    from numpy import zeros
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    arr_u      = zeros((k1,k2))
    arr_ux     = zeros((k1,k2))
    arr_uy     = zeros((k1,k2))
    #...
    arr_uxx    = zeros((k1,k2))
    arr_uyy    = zeros((k1,k2))
    #...
    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            # ...
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    
                    sxx = 0.0
                    syy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_0      = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                            bj_x      = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y      = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            
                            bj_xx     = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                            bj_yy     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                            # ...
                            coeff_u   = lcoeffs_u[il_1,il_2]
                            # ...
                            s        +=  coeff_u*bj_0
                            sx       +=  coeff_u*bj_x
                            sy       +=  coeff_u*bj_y
                            # ...
                            sxx      +=  coeff_u*bj_xx
                            syy      +=  coeff_u*bj_yy

                    arr_u[g1,g2]      = s
                    arr_ux[g1,g2]     = sx
                    arr_uy[g1,g2]     = sy
                    
                    arr_uxx[g1,g2]    = syy
                    arr_uyy[g1,g2]    = sxx

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1
                            # ...
                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2
                            # ...
                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bi_xx = basis_1[ie1, il_1, 2, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_yy = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 2, g2]
                                    
                                    #........
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bj_xx = basis_1[ie1, jl_1, 2, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_yy = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 2, g2]
                                    #........ 
                                    u    = arr_u[g1,g2] 
                                    ux   = arr_ux[g1,g2]
                                    uy   = arr_uy[g1,g2]
                    
                                    uxx  = arr_uxx[g1,g2]
                                    uyy  = arr_uyy[g1,g2]                                    
                                    #..
                                    R_1  = ( (3.*alpha/(2.*theta))*(1.-4.*theta*u*(1.-u)) + (1.-2.*u)*(uxx+uyy) ) * (bj_x1*bi_x1 + bj_x2 * bi_x2)
                                    R_2  = ( (-6.*alpha)*(1.-2.*u)*bi_0 - 2.*bi_0*(uxx+uyy) + (1.-2.*u)*(bi_xx+bi_yy) ) * (bj_x1*ux + bj_x2 * uy)
                                    #..
                                    R_3  = (bj_xx+bj_yy)*(bi_xx+bi_yy)*u*(1.-u)
                                    R_4  = (bj_xx+bj_yy)*(uxx+uyy)*(1.-2.*u)*bi_0
                                    # ...
                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    +=  bi_0 * bj_0 * wvol +  dt * ( R_1 + R_2 + R_3 + R_4 ) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================
#---2 : In adapted mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'float', 'float', 'float', 'double[:,:,:,:]')
def assemble_matrix_ex04(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, vector_u, vector_v, vector_w, vector_u1, vector_v1, dt, alpha, theta, matrix):

    # ... sizes
    from numpy import zeros
    k1 = weights_5.shape[1]
    k2 = weights_6.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p3+1))
    lcoeffs_v  = zeros((p4+1,p2+1))
    # ...
    lcoeffs_u1 = zeros((p1+1,p3+1))
    lcoeffs_v1 = zeros((p4+1,p2+1))
    #..
    arr_J_F1x  = zeros((k1,k2))
    arr_J_F1y  = zeros((k1,k2))
    arr_J_F2y  = zeros((k1,k2))
    arr_J_F2x  = zeros((k1,k2))
    #..
    arr_J_F1xx  = zeros((k1,k2))
    arr_J_F1xy  = zeros((k1,k2))
    arr_J_F1yy  = zeros((k1,k2))
    #..
    arr_J_F2xx  = zeros((k1,k2))
    arr_J_F2xy  = zeros((k1,k2))
    arr_J_F2yy  = zeros((k1,k2))
    #..
    J_mat       = zeros((k1,k2))
    # ...
    lcoeffs_w   = zeros((p5+1,p6+1))
    # ...
    arr_u       = zeros((k1,k2))
    arr_ux      = zeros((k1,k2))
    arr_uy      = zeros((k1,k2))
    #...
    arr_uxx     = zeros((k1,k2))
    arr_uyy     = zeros((k1,k2))
    #...
    arr_xp     = zeros((k1,k2))
    arr_yp     = zeros((k1,k2))
    # ... build matrices
    for ie1 in range(0, ne5):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        
        for ie2 in range(0, ne6):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]

            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_v1[ : , : ] = vector_v1[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_v[ : , : ] = vector_v[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1  = 0.0
                    xp1 = 0.0
                    F1x = 0.0
                    F1y = 0.0
                    F1xx= 0.0
                    F1xy= 0.0
                    F1yy= 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0     = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_3[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,1,g2]
                              
                              bj_xx    = basis_1[ie1,il_1,2,g1]*basis_3[ie2,il_2,0,g2]
                              bj_xy    = basis_1[ie1,il_1,1,g1]*basis_3[ie2,il_2,1,g2]
                              bj_yy    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,2,g2]

                              coeff_u  = lcoeffs_u[il_1,il_2]

                              x1      +=  coeff_u*bj_0
                              F1x     +=  coeff_u*bj_x
                              F1y     +=  coeff_u*bj_y
                              F1xx    +=  coeff_u*bj_xx
                              F1xy    +=  coeff_u*bj_xy
                              F1yy    +=  coeff_u*bj_yy

                              coeff_u1 = lcoeffs_u1[il_1,il_2]
                              
                              xp1     +=  coeff_u1*bj_0
                    x2  = 0.0
                    yp2 = 0.0
                    F2y = 0.0
                    F2x = 0.0
                    F2xx= 0.0
                    F2xy= 0.0
                    F2yy= 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x    = basis_4[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              bj_xx   = basis_4[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_xy   = basis_4[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              bj_yy   = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              
                              coeff_v = lcoeffs_v[il_1,il_2]
                              
                              x2     +=  coeff_v*bj_0
                              F2x    +=  coeff_v*bj_x
                              F2y    +=  coeff_v*bj_y
                              F2xx   +=  coeff_v*bj_xx
                              F2xy   +=  coeff_v*bj_xy
                              F2yy   +=  coeff_v*bj_yy

                              coeff_v1= lcoeffs_v1[il_1,il_2]
                              
                              yp2    +=  coeff_v1*bj_0
                    abs_mat          = abs(F1x*F2y-F1y**2)
                    arr_J_F2x[g1,g2] = F2x
                    arr_J_F2y[g1,g2] = F2y
                    arr_J_F1x[g1,g2] = F1x
                    arr_J_F1y[g1,g2] = F1y
                    #..
                    arr_J_F1xx[g1,g2] = F1xx
                    arr_J_F1xy[g1,g2] = F1xy
                    arr_J_F1yy[g1,g2] = F1yy
                    arr_J_F2xx[g1,g2] = F2xx
                    arr_J_F2xy[g1,g2] = F2xy
                    arr_J_F2yy[g1,g2] = F2yy
                    # ...
                    J_mat[g1,g2]      = abs_mat
                    # ...
                    arr_xp[g1,g2]      = (x1-xp1)/dt
                    arr_yp[g1,g2]      = (x2-yp2)/dt
                    # ...
                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    
                    sxx = 0.0
                    sxy = 0.0
                    syy = 0.0
                    for il_1 in range(0, p5+1):
                        for il_2 in range(0, p6+1):

                            bj_0      = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                            bj_x      = basis_5[ie1,il_1,1,g1]*basis_6[ie2,il_2,0,g2]
                            bj_y      = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,1,g2]
                            
                            bj_xx     = basis_5[ie1,il_1,2,g1]*basis_6[ie2,il_2,0,g2]
                            bj_yy     = basis_5[ie1,il_1,1,g1]*basis_6[ie2,il_2,1,g2]
                            bj_yy     = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,2,g2]
                            # ...
                            coeff_w   = lcoeffs_w[il_1,il_2]
                            # ...
                            s        +=  coeff_w*bj_0
                            sx       +=  coeff_w*bj_x
                            sy       +=  coeff_w*bj_y
                            # ...
                            sxx      +=  coeff_w*bj_xx
                            sxy      +=  coeff_w*bj_xy
                            syy      +=  coeff_w*bj_yy

                    arr_u[g1,g2]      = s
                    arr_ux[g1,g2]     = (F2y * sx - F1y * sy)/abs_mat
                    arr_uy[g1,g2]     = (F1x * sy - F1y * sx)/abs_mat
                    # ...
                    C1                = (F2y * F2xy + F1y * F1xy) * sx + (F1y * F1y  + F2y * F2y ) * sxx
                    C2                = (F2y * F1xy + F1y * F1xx) * sy + (F2y * F1y  + F1y * F1x ) * sxy
                    C3                = (F1y * arr_uy[g1,g2]  - F2y * arr_ux[g1,g2] ) * (F1xx*F2y+F1x*F2xy-2.*F1xy*F1y)
                    # ..
                    C4                = (F1y * F2yy + F1x * F1yy) * sx + (F1y * F2y  + F1x * F1y ) * sxy
                    C5                = (F1y * F1yy + F1x * F1xy) * sy + (F1y * F1y  + F1x * F1x ) * syy
                    C6                = (F1y * arr_ux[g1,g2]  - F1x * arr_uy[g1,g2] ) * (F1xy*F2y+F1x*F2yy-2.*F1yy*F1y)
                    # ...
                    arr_uxx[g1,g2]    = (C1 - C2 + C3)/abs_mat**2
                    arr_uyy[g1,g2]    = (C5 - C4 + C6)/abs_mat**2

            for il_1 in range(0, p5+1):
                for il_2 in range(0, p6+1):
                    for jl_1 in range(0, p5+1):
                        for jl_2 in range(0, p6+1):
                            i1 = i_span_5 - p5 + il_1
                            j1 = i_span_5 - p5 + jl_1

                            i2 = i_span_6 - p6 + il_2
                            j2 = i_span_6 - p6 + jl_2

                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    F2x     = arr_J_F2x[g1,g2] 
                                    F2y     = arr_J_F2y[g1,g2] 
                                    F1x     = arr_J_F1x[g1,g2] 
                                    F1y     = arr_J_F1y[g1,g2]
                                    #..
                                    F1xx    = arr_J_F1xx[g1,g2] 
                                    F1xy    = arr_J_F1xy[g1,g2] 
                                    F1yy    = arr_J_F1yy[g1,g2] 
                                    F2xx    = arr_J_F2xx[g1,g2]
                                    F2xy    = arr_J_F2xy[g1,g2]
                                    F2yy    = arr_J_F2yy[g1,g2]
                                    # ...
                                    abs_mat = J_mat[g1,g2]                                
                                    #...
                                    bi_0    = basis_5[ie1, il_1, 0, g1] * basis_6[ie2, il_2, 0, g2]
                                    bi_x1   = basis_5[ie1, il_1, 1, g1] * basis_6[ie2, il_2, 0, g2]
                                    bi_x2   = basis_5[ie1, il_1, 0, g1] * basis_6[ie2, il_2, 1, g2]
                                    bi_xx   = basis_5[ie1, il_1, 2, g1] * basis_6[ie2, il_2, 0, g2]
                                    bi_xy   = basis_5[ie1, il_1, 1, g1] * basis_6[ie2, il_2, 1, g2]
                                    bi_yy   = basis_5[ie1, il_1, 0, g1] * basis_6[ie2, il_2, 2, g2]
                                    # ... 
                                    bj_0    = basis_5[ie1, jl_1, 0, g1] * basis_6[ie2, jl_2, 0, g2]
                                    bj_x1   = basis_5[ie1, jl_1, 1, g1] * basis_6[ie2, jl_2, 0, g2]
                                    bj_x2   = basis_5[ie1, jl_1, 0, g1] * basis_6[ie2, jl_2, 1, g2]
                                    bj_xx   = basis_5[ie1, jl_1, 2, g1] * basis_6[ie2, jl_2, 0, g2]
                                    bj_xy   = basis_5[ie1, jl_1, 1, g1] * basis_6[ie2, jl_2, 1, g2]
                                    bj_yy   = basis_5[ie1, jl_1, 0, g1] * basis_6[ie2, jl_2, 2, g2]
                                    # ... 
                                    bi_x    = (F2y * bi_x1 - F1y * bi_x2)/abs_mat
                                    bi_y    = (F1x * bi_x2 - F1y * bi_x1)/abs_mat

                                    # ...
                                    C1      = (F2y * F2xy + F1y * F1xy) * bi_x1 + (F1y * F1y  + F2y * F2y ) * bi_xx
                                    C2      = (F2y * F1xy + F1y * F1xx) * bi_x2 + (F2y * F1y  + F1y * F1x ) * bi_xy
                                    C3      = (F1y * bi_y  - F2y * bi_x ) * (F1xx*F2y+F1x*F2xy-2.*F1xy*F1y)
                                    # ..
                                    C4      = (F1y * F2yy + F1x * F1yy) * bi_x1 + (F1y * F2y  + F1x * F1y ) * bi_xy
                                    C5      = (F1y * F1yy + F1x * F1xy) * bi_x2 + (F1y * F1y  + F1x * F1x ) * bi_yy
                                    C6      = (F1y * bi_x  - F1x * bi_y ) * (F1xy*F2y+F1x*F2yy-2.*F1yy*F1y)
                                    #...
                                    bi_uxx  = (C1 - C2 + C3)/abs_mat**2
                                    bi_uyy  = (C5 - C4 + C6)/abs_mat**2
                                    # ... ---
                                    bj_x    = (F2y * bj_x1 - F1y * bj_x2)/abs_mat
                                    bj_y    = (F1x * bj_x2 - F1y * bj_x1)/abs_mat
                                    
                                    # ...
                                    C1      = (F2y * F2xy + F1y * F1xy) * bj_x1 + (F1y * F1y  + F2y * F2y ) * bj_xx
                                    C2      = (F2y * F1xy + F1y * F1xx) * bj_x2 + (F2y * F1y  + F1y * F1x ) * bj_xy
                                    C3      = (F1y * bj_y  - F2y * bj_x ) * (F1xx*F2y+F1x*F2xy-2.*F1xy*F1y)
                                    # ..
                                    C4      = (F1y * F2yy + F1x * F1yy) * bj_x1 + (F1y * F2y  + F1x * F1y ) * bj_xy
                                    C5      = (F1y * F1yy + F1x * F1xy) * bj_x2 + (F1y * F1y  + F1x * F1x ) * bj_yy
                                    C6      = (F1y * bj_x  - F1x * bj_y ) * (F1xy*F2y+F1x*F2yy-2.*F1yy*F1y)
                                    #...
                                    bj_uxx  = (C1 - C2 + C3)/abs_mat**2
                                    bj_uyy  = (C5 - C4 + C6)/abs_mat**2

                                    # ---
                                    wvol    = weights_5[ie1, g1] * weights_6[ie2, g2] * abs_mat
                                    #........ 
                                    u       = arr_u[g1,g2] 
                                    ux      = arr_ux[g1,g2]
                                    uy      = arr_uy[g1,g2]
                                    uxx     = arr_uxx[g1,g2]
                                    uyy     = arr_uyy[g1,g2]                                    
                                    # ...
                                    xp1   = arr_xp[g1,g2]
                                    xp2   = arr_yp[g1,g2]
                                    #..
                                    R_1  = ( (3.*alpha/(2.*theta))*(1.-4.*theta*u*(1.-u)) + (1.-2.*u)*(uxx+uyy)       ) * (bj_x*bi_x + bj_y * bi_y)
                                    R_2  = ( (-6.*alpha)*(1.-2.*u)*bi_0 - 2.*bi_0*(uxx+uyy) + (1.-2.*u)*(bi_uxx+bi_uyy) ) * (bj_x*ux + bj_y * uy)
                                    #..
                                    R_3  = (bj_uxx+bj_uyy)*(bi_uxx+bi_uyy)*u*(1.-u)
                                    R_4  = (bj_uxx+bj_uyy)*(uxx+uyy)*(1.-2.*u)*bi_0
                                    #...
                                    conse = -1.*( xp1*bi_x + xp2*bi_y) * bj_0 * wvol
                                    v    += bi_0 * bj_0 * wvol +  dt * ( R_1 + R_2 + R_3 + R_4 + conse) * wvol

                            matrix[p5+i1, p6+i2, p5+j1-i1, p6+j2-i2]  += v
    # ...
    
#==============================================================================
#---2 : In usiform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, dt, alpha, theta, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import arctan2
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #...
    lcoeffs_u  = zeros((p1+1,p2+1))
    lcoeffs_w  = zeros((p1+1,p2+1))
    # ...
    arr_ut     = zeros((k1,k2))
    arr_u      = zeros((k1,k2))
    arr_ux     = zeros((k1,k2))
    arr_uy     = zeros((k1,k2))
    # ...
    arr_uxx    = zeros((k1,k2))
    arr_uyy    = zeros((k1,k2))
    # +++
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    st  = 0.0
                    #...
                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    sxx = 0.0
                    syy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # +++
                              bj_xx   = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy   = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              # +++
                              coeff_u = lcoeffs_u[il_1,il_2]
                              # +++
                              s      +=  coeff_u*bj_0
                              sx     +=  coeff_u*bj_x
                              sy     +=  coeff_u*bj_y
                              
                              sxx    +=  coeff_u*bj_xx
                              syy    +=  coeff_u*bj_yy
                              # +++
                              coeff_w = lcoeffs_w[il_1,il_2]
                              # +++
                              st     +=  coeff_w*bj_0


                    arr_ut[g1,g2]     = st
                    arr_u[g1,g2]      = s
                    arr_ux[g1,g2]     = sx
                    arr_uy[g1,g2]     = sy
                    
                    arr_uxx[g1,g2]    = syy
                    arr_uyy[g1,g2]    = sxx
                    
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):

                            bi_0    = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,0,g2]
                            bi_x    = basis_1[ie1, il_1,1,g1] * basis_2[ie2,il_2,0,g2]
                            bi_y    = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,1,g2]
                            # +++
                            bi_xx   = basis_1[ie1, il_1,2,g1] * basis_2[ie2,il_2,0,g2]
                            bi_yy   = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,2,g2]
                            
                            wvol    = weights_1[ie1, g1] * weights_2[ie2, g2]
                            #........ 
                            ut      = arr_ut[g1,g2] 
                            u       = arr_u[g1,g2] 
                            ux      = arr_ux[g1,g2]
                            uy      = arr_uy[g1,g2]
                    
                            uxx     = arr_uxx[g1,g2]
                            uyy     = arr_uyy[g1,g2] 
                            #...
                            R_1     = ((3.*alpha/(2.*theta))*(1. - 4.*theta*u*(1.-u) ) + (1.-2.*u) * (uxx+uyy) ) * (bi_x * ux + bi_y * uy) 
                            
                            R_2     = u * (1. - u) * (uxx + uyy) * (bi_xx + bi_yy) 

                            v      +=  bi_0 * u * wvol + dt * (R_1 * wvol + R_2 * wvol) - 1.* bi_0 * ut * wvol

                    rhs[i1+p1,i2+p2] += v
    # ...


#==============================================================================
#---2 : In adapted mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_ex02(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, vector_u, vector_v, vector_w, vector_z, vector_u1, vector_v1, dt, alpha, theta, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import arctan2
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    lcoeffs_u  = zeros((p1+1,p3+1))
    lcoeffs_v  = zeros((p4+1,p2+1))
    # ...
    lcoeffs_u1 = zeros((p1+1,p3+1))
    lcoeffs_v1 = zeros((p4+1,p2+1))
    #..
    arr_J_F1x  = zeros((k1,k2))
    arr_J_F1y  = zeros((k1,k2))
    arr_J_F2y  = zeros((k1,k2))
    arr_J_F2x  = zeros((k1,k2))
    #..
    arr_J_F1xx  = zeros((k1,k2))
    arr_J_F1xy  = zeros((k1,k2))
    arr_J_F1yy  = zeros((k1,k2))
    #..
    arr_J_F2xx  = zeros((k1,k2))
    arr_J_F2xy  = zeros((k1,k2))
    arr_J_F2yy  = zeros((k1,k2))
    # ...
    J_mat       = zeros((k1,k2))
    # ...
    lcoeffs_w   = zeros((p5+1,p6+1))
    lcoeffs_z   = zeros((p5+1,p6+1))
    # ...
    arr_ut      = zeros((k1,k2))
    arr_u       = zeros((k1,k2))
    arr_ux      = zeros((k1,k2))
    arr_uy      = zeros((k1,k2))
    # ...
    arr_uxx     = zeros((k1,k2))
    arr_uyy     = zeros((k1,k2))
    # ...
    arr_xp     = zeros((k1,k2))
    arr_yp     = zeros((k1,k2))
    # ... build rhs
    for ie1 in range(0, ne5):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        
        for ie2 in range(0, ne6):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]

            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_v1[ : , : ] = vector_v1[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_v[ : , : ] = vector_v[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            lcoeffs_z[ : , : ] = vector_z[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1  = 0.0
                    xp1 = 0.0
                    F1x = 0.0
                    F1y = 0.0
                    F1xx= 0.0
                    F1xy= 0.0
                    F1yy= 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0     = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_3[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,1,g2]
                              
                              bj_xx    = basis_1[ie1,il_1,2,g1]*basis_3[ie2,il_2,0,g2]
                              bj_xy    = basis_1[ie1,il_1,1,g1]*basis_3[ie2,il_2,1,g2]
                              bj_yy    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,2,g2]

                              coeff_u  = lcoeffs_u[il_1,il_2]

                              x1      +=  coeff_u*bj_0
                              F1x     +=  coeff_u*bj_x
                              F1y     +=  coeff_u*bj_y
                              F1xx    +=  coeff_u*bj_xx
                              F1xy    +=  coeff_u*bj_xy
                              F1yy    +=  coeff_u*bj_yy

                              coeff_u1 = lcoeffs_u1[il_1,il_2]

                              xp1     +=  coeff_u1*bj_0
                    x2  = 0.0
                    yp2 = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    F2xx= 0.0
                    F2xy= 0.0
                    F2yy= 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x    = basis_4[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              bj_xx   = basis_4[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_xy   = basis_4[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]
                              bj_yy   = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              
                              coeff_v = lcoeffs_v[il_1,il_2]
                              
                              x2     +=  coeff_v*bj_0
                              F2y    +=  coeff_v*bj_y
                              F2x    +=  coeff_v*bj_x
                              F2xx   +=  coeff_v*bj_xx
                              F2xy   +=  coeff_v*bj_xy
                              F2yy   +=  coeff_v*bj_yy
 
                              coeff_v1= lcoeffs_v1[il_1,il_2]
                              
                              yp2    +=  coeff_v1*bj_0

                    abs_mat           = abs(F1x*F2y-F1y*F2x)
                    arr_J_F2y[g1,g2]  = F2y
                    arr_J_F2x[g1,g2]  = F2x
                    arr_J_F1x[g1,g2]  = F1x
                    arr_J_F1y[g1,g2]  = F1y
                    #..
                    arr_J_F1xx[g1,g2] = F1xx
                    arr_J_F1xy[g1,g2] = F1xy
                    arr_J_F1yy[g1,g2] = F1yy
                    arr_J_F2xx[g1,g2] = F2xx
                    arr_J_F2xy[g1,g2] = F2xy
                    arr_J_F2yy[g1,g2] = F2yy
                    # ...
                    J_mat[g1,g2]      = abs_mat
                    # ...
                    arr_xp[g1, g2] = (x1-xp1)/dt
                    arr_yp[g1, g2] = (x2-yp2)/dt
                    # ...
                    st  = 0.0
                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    
                    sxx = 0.0
                    sxy = 0.0
                    syy = 0.0
                    for il_1 in range(0, p5+1):
                        for il_2 in range(0, p6+1):

                            bj_0      = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                            bj_x      = basis_5[ie1,il_1,1,g1]*basis_6[ie2,il_2,0,g2]
                            bj_y      = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,1,g2]
                            
                            bj_xx     = basis_5[ie1,il_1,2,g1]*basis_6[ie2,il_2,0,g2]
                            bj_xy     = basis_5[ie1,il_1,1,g1]*basis_6[ie2,il_2,1,g2]
                            bj_yy     = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,2,g2]
                            # ...
                            coeff_w   = lcoeffs_w[il_1,il_2]
                            # ...
                            s        +=  coeff_w*bj_0
                            sx       +=  coeff_w*bj_x
                            sy       +=  coeff_w*bj_y
                            # ...
                            sxx      +=  coeff_w*bj_xx
                            sxy      +=  coeff_w*bj_xy
                            syy      +=  coeff_w*bj_yy
                            # ...
                            coeff_z   = lcoeffs_z[il_1,il_2]
                            # ...
                            st        +=  coeff_z*bj_0
                            
                    arr_u[g1,g2]      = s
                    arr_ut[g1,g2]     = st
                    arr_ux[g1,g2]     = (F2y * sx - F1y * sy)/abs_mat
                    arr_uy[g1,g2]     = (F1x * sy - F1y * sx)/abs_mat
                    # ...
                    C1                = (F2y * F2xy + F1y * F1xy) * sx + (F1y * F1y  + F2y * F2y ) * sxx
                    C2                = (F2y * F1xy + F1y * F1xx) * sy + (F2y * F1y  + F1y * F1x ) * sxy
                    C3                = (F1y * arr_uy[g1,g2]  - F2y * arr_ux[g1,g2] ) * (F1xx*F2y+F1x*F2xy-2.*F1xy*F1y)
                    # ..
                    C4                = (F1y * F2yy + F1x * F1yy) * sx + (F1y * F2y  + F1x * F1y ) * sxy
                    C5                = (F1y * F1yy + F1x * F1xy) * sy + (F1y * F1y  + F1x * F1x ) * syy
                    C6                = (F1y * arr_ux[g1,g2]  - F1x * arr_uy[g1,g2] ) * (F1xy*F2y+F1x*F2yy-2.*F1yy*F1y)
                    # ...
                    arr_uxx[g1,g2]    = (C1 - C2 + C3)/abs_mat**2
                    arr_uyy[g1,g2]    = (C5 - C4 + C6)/abs_mat**2
                    
            for il_1 in range(0, p5+1):
                for il_2 in range(0, p6+1):
                    i1 = i_span_5 - p5 + il_1
                    i2 = i_span_6 - p6 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            F2y     = arr_J_F2y[g1,g2]
                            F2x     = arr_J_F2x[g1,g2]
                            F1x     = arr_J_F1x[g1,g2] 
                            F1y     = arr_J_F1y[g1,g2]
                            #..
                            F1xx    = arr_J_F1xx[g1,g2] 
                            F1xy    = arr_J_F1xy[g1,g2] 
                            F1yy    = arr_J_F1yy[g1,g2] 
                            F2xx    = arr_J_F2xx[g1,g2]
                            F2xy    = arr_J_F2xy[g1,g2]
                            F2yy    = arr_J_F2yy[g1,g2]
                            # ...
                            abs_mat = J_mat[g1,g2] 
                            #...
                            bi_0    = basis_5[ie1, il_1,0,g1] * basis_6[ie2,il_2,0,g2]
                            bi_x1   = basis_5[ie1, il_1,1,g1] * basis_6[ie2,il_2,0,g2]
                            bi_x2   = basis_5[ie1, il_1,0,g1] * basis_6[ie2,il_2,1,g2]
                            # +++
                            bi_xx   = basis_5[ie1, il_1,2,g1] * basis_6[ie2,il_2,0,g2]
                            bi_xy   = basis_5[ie1, il_1,1,g1] * basis_6[ie2,il_2,1,g2]
                            bi_yy   = basis_5[ie1, il_1,0,g1] * basis_6[ie2,il_2,2,g2]
                            # ... ---
                            bi_x    = (F2y * bi_x1 - F1y * bi_x2)/abs_mat
                            bi_y    = (F1x * bi_x2 - F1y * bi_x1)/abs_mat
                            # ...
                            C1      = (F2y * F2xy + F1y * F1xy) * bi_x1 + (F1y * F1y  + F2y * F2y ) * bi_xx
                            C2      = (F2y * F1xy + F1y * F1xx) * bi_x2 + (F2y * F1y  + F1y * F1x ) * bi_xy
                            C3      = (F1y * bi_y  - F2y * bi_x ) * (F1xx*F2y+F1x*F2xy-2.*F1xy*F1y)
                            # ..
                            C4      = (F1y * F2yy + F1x * F1yy) * bi_x1 + (F1y * F2y  + F1x * F1y ) * bi_xy
                            C5      = (F1y * F1yy + F1x * F1xy) * bi_x2 + (F1y * F1y  + F1x * F1x ) * bi_yy
                            C6      = (F1y * bi_x  - F1x * bi_y ) * (F1xy*F2y+F1x*F2yy-2.*F1yy*F1y)
                            #...
                            bi_uxx  = (C1 - C2 + C3)/abs_mat**2
                            bi_uyy  = (C5 - C4 + C6)/abs_mat**2
                            
                            wvol    = weights_1[ie1, g1] * weights_2[ie2, g2] * abs_mat
                            #........ 
                            u       = arr_u[g1,g2] 
                            ut      = arr_ut[g1,g2] 
                            ux      = arr_ux[g1,g2]
                            uy      = arr_uy[g1,g2]
                    
                            uxx     = arr_uxx[g1,g2]
                            uyy     = arr_uyy[g1,g2] 
                            # ...
                            xp1   = arr_xp[g1, g2]
                            xp2   = arr_yp[g1, g2]
                            #...
                            R_1     = ((3.*alpha/(2.*theta))*(1. - 4.*theta*u*(1.-u) ) + (1.-2.*u) * (uxx+uyy) ) * (bi_x * ux + bi_y * uy) 
                            
                            R_2     = u * (1. - u) * (uxx + uyy) * (bi_uxx + bi_uyy)
                            # ... 
                            conse   = -1. * ( xp1*ux + xp2*uy) * bi_0 * wvol
                            v      +=  bi_0 * u * wvol + dt* (R_1 + R_2 + conse) * wvol - 1.* bi_0 * ut * wvol
                    rhs[i1+p5,i2+p6] += v
    # ...

#==============================================================================
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, alpha, theta, rhs):

    from numpy import sin
    from numpy import cos
    from numpy import pi
    from numpy import sqrt
    from numpy import exp
    from numpy import log
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    t = 0.
    lcoeffs_u  = zeros((p1+1,p2+1))
    lvalues_u  = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]  = 0.0
            lvalues_ux[ : , : ] = 0.0
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]
                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2]
                            db2  = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_u*b1*b2
                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]

                    # ... Test 0
                    #u   =   exp(-500*(((x-0.5)/0.4)**2+((y-0.5)/0.4)**2-0.01-t)**2)
                    #ux  =  -19531.25*(4*x - 2.0)*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)*exp(-19531.25*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)**2)  
                    #uy  = -19531.25*(4*y - 2.0)*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)*exp(-19531.25*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)**2)

                    # ... Test 1
                    u   =   1./(1.+exp((x + y  - 0.25 - t)/0.01) )
                    ux  =  -1.3887943864964e-9*exp(-100.0*t + 100.0*x + 100.0*y)/(1.3887943864964e-11*exp(-100.0*t + 100.0*x + 100.0*y) + 1.0)**2
                    uy  =  -1.3887943864964e-9*exp(-100.0*t + 100.0*x + 100.0*y)/(1.3887943864964e-11*exp(-100.0*t + 100.0*x + 100.0*y) + 1.0)**2

                    # ... Test 2
                    #u    = exp(-500*(y-0.5-0.25*sin(2*pi*x)*sin(0.6*pi*(1.+50.*t)))**2)
                    #ux   = 500.0*pi*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)*exp(-500*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)**2)*sin(pi*(30.0*t + 0.6))*cos(2*pi*x)
                    #uy   = (-1000*y + 250.0*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) + 500.0)*exp(-500*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)**2)
                    
                    #+--------
                    uh   = lvalues_u[g1,g2]
                    uhx  = lvalues_ux[g1,g2]
                    uhy  = lvalues_uy[g1,g2]
                    #+++++++++
                    if uh > 1. :
                        uh = 0.9999
                    if uh < 0. :
                         uh = 0.0001    
                    w   += ( uh*log(uh) + (1.-uh)*log(1.-uh) + 2.*theta*uh*(1.-uh)+ theta/(3.*alpha)*(uhx**2 + uhy**2) )* wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = norm_H1
    rhs[p1, p2]   = norm_l2
    rhs[p1, p2+1] = norm_H1
    # ...


#==============================================================================
#---2 : In adapted mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'double[:,:]')
def assemble_norm_ex02(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, vector_u, vector_w, vector_z, alpha, theta, rhs):

    from numpy import exp
    from numpy import log
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    
    # ... sizes
    k1          = weights_1.shape[1]
    k2          = weights_2.shape[1]
    # ...
    lcoeffs_u   = zeros((p1+1,p3+1))
    lcoeffs_w   = zeros((p4+1,p2+1))
    lcoeffs_z   = zeros((p5+1,p6+1))
    
    lvalues_u   = zeros((k1, k2))
    lvalues_ux  = zeros((k1, k2))
    lvalues_uy  = zeros((k1, k2))
    # ...
    lvalues_u1  = zeros((k1, k2))
    lvalues_u1x = zeros((k1, k2))
    lvalues_u1y = zeros((k1, k2))
    lvalues_u2  = zeros((k1, k2))
    #lvalues_u2x = zeros((k1, k2))
    lvalues_u2y = zeros((k1, k2))
    # ..
    t = 0.
    # ..
    norm_l2      = 0.                                
    norm_H1      = 0.
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p3+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_3[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_3[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1[g1,g2]  += coeff_u*b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2
            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 
                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2
            lvalues_u[ : , : ]      = 0.0
            lvalues_ux[ : , : ]     = 0.0
            lvalues_uy[ : , : ]     = 0.0

            lcoeffs_z[ : , : ] = vector_z[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            for il_1 in range(0, p5+1):
                for il_2 in range(0, p6+1):
                    coeff_z = lcoeffs_z[il_1,il_2]

                    for g1 in range(0, k1):
                        b1 = basis_5[ie1,il_1,0,g1]
                        db1 = basis_5[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_6[ie2,il_2,0,g2]
                            db2  = basis_6[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_z*b1*b2
                            lvalues_ux[g1,g2] += coeff_z*db1*b2
                            lvalues_uy[g1,g2] += coeff_z*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x    = lvalues_u1[g1,g2]
                    y    = lvalues_u2[g1,g2]

                    sxx  = lvalues_u1x[g1,g2]
                    syy  = lvalues_u2y[g1,g2]
                    sxy  = lvalues_u1y[g1,g2]

                    # ---
                    # ... Test 0
                    #u   =   exp(-500*(((x-0.5)/0.4)**2+((y-0.5)/0.4)**2-0.01-t)**2)
                    #ux  =  -19531.25*(4*x - 2.0)*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)*exp(-19531.25*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)**2)  
                    #uy  = -19531.25*(4*y - 2.0)*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)*exp(-19531.25*(-0.16*t + (x - 0.5)**2 + (y - 0.5)**2 - 0.0016)**2)

                    # ... Test 1
                    u   =   1./(1.+exp((x + y  - 0.25 - t)/0.01) )
                    ux  =  -1.3887943864964e-9*exp(-100.0*t + 100.0*x + 100.0*y)/(1.3887943864964e-11*exp(-100.0*t + 100.0*x + 100.0*y) + 1.0)**2
                    uy  =  -1.3887943864964e-9*exp(-100.0*t + 100.0*x + 100.0*y)/(1.3887943864964e-11*exp(-100.0*t + 100.0*x + 100.0*y) + 1.0)**2

                    # ... Test 2
                    #u    = exp(-500*(y-0.5-0.25*sin(2*pi*x)*sin(0.6*pi*(1.+50.*t)))**2)
                    #ux   = 500.0*pi*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)*exp(-500*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)**2)*sin(pi*(30.0*t + 0.6))*cos(2*pi*x)
                    #uy   = (-1000*y + 250.0*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) + 500.0)*exp(-500*(y - 0.25*sin(2*pi*x)*sin(pi*(30.0*t + 0.6)) - 0.5)**2)
                                        
                    uh    = lvalues_u[g1,g2]
                    uhx   = lvalues_ux[g1,g2]
                    uhy   = lvalues_uy[g1,g2]

                    J_mat = abs(sxx*syy-sxy**2)
                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2] * J_mat

                    #+++++++++
                    if uh > 1. :
                        uh = 0.9999
                    if uh < 0. :
                         uh = 0.0001    
                    w   += ( uh*log(uh) + (1.-uh)*log(1.-uh) + 2.*theta*uh*(1.-uh)+ theta/(3.*alpha)*(((syy*uhx - sxy*uhy)/J_mat)**2 + ((-sxy*uhx + sxx*uhy)/J_mat)**2) )* wvol
                    
                    v  += (u-uh)**2 * wvol
                    #w  += ((ux- (syy*uhx - sxy*uhy)/J_mat)**2+ (uy- (-sxy*uhx + sxx*uhy)/J_mat)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = norm_H1

    rhs[p5,p6]   = norm_l2
    rhs[p5,p6+1] = norm_H1
    # ...
