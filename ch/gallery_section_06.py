__all__ = ['assemble_vector_ex01',
           'assemble_norm_ex01'
]
from pyccel.decorators import types



# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v   += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v   += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...

@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex01(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]  
            i_span_2 = spans_2[ie1]      
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis_1[ie1, il_1, 1, g1]
                                    bj_0 = basis_2[ie1, il_2, 0, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v   += bi_x * bj_0 * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...


@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex02(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis_1[ie1, il_1, 0, g1]
                                    bj_x = basis_2[ie1, il_2, 1, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v   += bi_0 * bj_x * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...

#==============================================================================
# Assembles a rhs at t_0
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_ex00(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, vector_z, epsilon, gamma, dt, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_z  = zeros((p3+1,p4+1))
    # ...
    dens_val   = zeros((k1,k2))
    # ...coefficient of normalisation
    crho      = 0.0
    for ie1 in range(0, ne1):
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne2):
            i_span_4 = spans_4[ie2]
            
            lcoeffs_z[ : , : ]  =  vector_z[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_3[ie1, g1] * weights_4[ie2, g2]

                    u_p        = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              bi_0      = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]

                              # ...
                              coef_z    = lcoeffs_z[il_1,il_2]
                              u_p      +=  coef_z*bi_0
                    # .. 
                    crho += abs(u_p) * wvol
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_4 = spans_4[ie2]
            
            lcoeffs_z[ : , : ]  =  vector_z[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    u_p        = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              coef_z    = lcoeffs_z[il_1,il_2]
                      
                              bi_0      = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                              # ...
                              u_p      +=  coef_z*bi_0
                    dens_val[g1, g2] = u_p
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            # .. 
                            rho       = (1. + abs(dens_val[g1, g2]))/(1.+crho)
                            sx        = points_1[ie1, g1]
                            sy        = points_2[ie2, g2]
                            # ...
                            u         = (sx**2+sy**2)*0.5
                            laplace_u = 2.0
                            mae_rho   = sqrt(rho)

                            v += ( epsilon*(u - gamma*2.0) + dt*mae_rho)* bi_0 * wvol

                    rhs[i1+p1,i2+p2] += v
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  x1       =  points_1[ie1, g1]
                  
                  vx_0    += bi_0*0.0 * wleng_x
                  vx_1    += bi_0*1. * wleng_x

           rhs[i1+p1,0+p2]       += epsilon*gamma*vx_0
           rhs[i1+p1,ne2+2*p2-1] += epsilon*gamma*vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  wleng_y =  weights_2[ie2, g2]
                  x2      =  points_2[ie2, g2]
                           
                  vy_0   += bi_0* 0.0 * wleng_y
                  vy_1   += bi_0*1. * wleng_y

           rhs[0+p1,i2+p2]       += epsilon*gamma*vy_0
           rhs[ne1-1+2*p1,i2+p2] += epsilon*gamma*vy_1
    # ...

    

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, knots_1, knots_2, knots_3, knots_4, vector_u, vector_z, epsilon, gamma, dt, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1          = weights_1.shape[1]
    k2          = weights_2.shape[1]
    lcoeffs_u   = zeros((p1+1,p2+1))
    lcoeffs_z   = zeros((p3+1,p4+1))

    lvalues_u   = zeros((k1, k2))
    lvalues_D   = zeros((k1, k2))
    lvalues_rho = zeros((k1, k2))
    # ...
    # ..
    points1    = zeros((ne1*ne2, k1*k2))
    points2    = zeros((ne1*ne2, k1*k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              sx     += coeff_u * bj_x
                              sy     += coeff_u * bj_y
                              
                    points1[ie2+ne2*ie1, g2+k2*g1] = sx
                    points2[ie2+ne2*ie1, g2+k2*g1] = sy

    #---Computes All basis in a new points
    nders          = 1
    degree         = p3
    #..
    ne, nq         = points1.shape
    xx             = zeros(nq)

    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis3         = zeros( (ne, degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points1[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_3)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_3[low ]: 
                 span = low
            if xq >= knots_3[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_3[span] or xq >= knots_3[span+1]:
                 if xq < knots_3[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_3[span-j]
                right[j] = knots_3[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis3[ie,:,0,iq] = ders[0,:]

    degree         = p4
    #...
    basis4         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points2[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_4)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_4[low ]: 
                 span = low
            if xq >= knots_4[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_4[span] or xq >= knots_4[span+1]:
                 if xq < knots_4[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_4[span-j]
                right[j] = knots_4[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis4[ie,:,0,iq] = ders[0,:]
    # ...coefficient of normalisation
    crho      = 0.0
    for ie1 in range(0, ne3):
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne4):
            i_span_4 = spans_4[ie2]
            
            lcoeffs_z[ : , : ]  =  vector_z[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_3[ie1, g1] * weights_4[ie2, g2]

                    u_p        = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                          
                              coef_z = lcoeffs_z[il_1,il_2]
                              #...
                              bi_0   = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                              #...
                              u_p   += coef_z*bi_0

                    crho += wvol * abs(u_p)

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    s   = 0.0
                    sxx = 0.0
                    syy = 0.0
                    sxy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0  = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]

                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              s    +=  coeff_u*bj_0

                              sxx  +=  coeff_u*bj_xx
                              syy  +=  coeff_u*bj_yy
                              sxy  +=  coeff_u*bj_xy
                              
                    x1   = points1[ie2+ne2*ie1, g2+k2*g1]
                    x2   = points2[ie2+ne2*ie1, g2+k2*g1]
                    #... We compute firstly the span in new adapted points
                    xq        = x1
                    degree    = p3
                    low       = degree
                    high      = len(knots_3)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_3[low ]: 
                         span = low
                    if xq >= knots_3[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_3[span] or xq >= knots_3[span+1]:
                         if xq < knots_3[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_3    = span
                    #...                    
                    xq        = x2
                    degree    = p4
                    low       = degree
                    high      = len(knots_4)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_4[low ]: 
                         span = low
                    if xq >= knots_4[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_4[span] or xq >= knots_4[span+1]:
                         if xq < knots_4[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_4    = span        

                    #------------------   
                    lcoeffs_z[ : , : ]  =  vector_z[span_3 : span_3+p3+1, span_4 : span_4+p4+1]
                    
                    u_p    = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              coef_z    = lcoeffs_z[il_1,il_2]
                              bi_0      = basis3[ie2+ne2*ie1,il_1,0,g2+k2*g1]*basis4[ie2+ne2*ie1,il_2,0,g2+k2*g1]
                              # ...
                              u_p       +=  coef_z*bi_0
                    # .. butterfluy
                    rho   = 1+1.*abs(u_p)

                    #...
                    rho                    = rho/(1.+1.*crho)
                    lvalues_u[g1,g2]       = s
                    lvalues_D[g1,g2]       = sxx+syy
                    lvalues_rho[g1,g2]     = sqrt(rho*abs(sxx*syy-sxy**2))

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u         = lvalues_u[g1,g2]
                            mae_rho   = lvalues_rho[g1,g2]
                            laplace_u = lvalues_D[g1,g2]

                            v += bi_0 *( epsilon*(u - gamma*laplace_u) + dt*mae_rho )* wvol

                    rhs[i1+p1,i2+p2] += v  
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  x1       =  points_1[ie1, g1]
                  
                  vx_0    += bi_0*0.0 * wleng_x
                  vx_1    += bi_0*1. * wleng_x

           rhs[i1+p1,0+p2]       += epsilon*gamma*vx_0
           rhs[i1+p1,ne2+2*p2-1] += epsilon*gamma*vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  wleng_y =  weights_2[ie2, g2]
                  x2      =  points_2[ie2, g2]
                           
                  vy_0   += bi_0* 0.0 * wleng_y
                  vy_1   += bi_0*1. * wleng_y

           rhs[0+p1,i2+p2]       += epsilon*gamma*vy_0
           rhs[ne1-1+2*p1,i2+p2] += epsilon*gamma*vy_1
    # ...

# ...
