__all__ = ['assemble_vector_ex01mae']

#---Assemble rhs of Picard-Monge-Ampere equation
# with density function equal to sqrt(1.+ |grad(u_h)(F)|) 

#==============================================================================
from pyccel.decorators import types

#==============================================================================
#---Assemble rhs of Mixed-BFO-Picard-Monge-Ampere equation
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int','int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:]')
def assemble_vector_ex01mae(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, knots_1, knots_2, knots_3, knots_4, knots_5, knots_6, vector_u, vector_w, vector_z, intensity, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p3+1))
    lcoeffs_w  = zeros((p4+1,p2+1))
    lcoeffs_u1 = zeros((p1+1,p3+1))
    lcoeffs_w1 = zeros((p4+1,p2+1))
    lcoeffs_z  = zeros((p5+1,p6+1))
    
    lvalues_u  = zeros((k1, k2))

    points1    = zeros((ne1*ne2, k1*k2))
    points2    = zeros((ne1*ne2, k1*k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              coeff_u = lcoeffs_u[il_1,il_2]

                              sx     +=  coeff_u*bj_0
                    sy = 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              coeff_w = lcoeffs_w[il_1,il_2]

                              sy     += coeff_w*bj_0
                    points1[ie2+ne2*ie1, g2+k2*g1] = sx
                    points2[ie2+ne2*ie1, g2+k2*g1] = sy

    #--Computes All basis in a new points
    nders          = 2
    degree         = p5
    #..
    ne, nq         = points1.shape
    xx             = zeros(nq)

    left           = empty( degree )
    right          = empty( degree )
    ndu            = empty( (degree+1, degree+1) )
    a              = empty( (       2, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis1         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points1[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_5)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_5[low ]: 
                 span = low
            if xq >= knots_5[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots_5[span] or xq >= knots_5[span+1]:
                 if xq < knots_5[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_5[span-j]
                right[j] = knots_5[span+1+j] - xq
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
            # Multiply derivatives by correct factors
            r = degree
            ders[1,:] = ders[1,:] * r
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis1[ie,:,0,iq] = ders[0,:]
            basis1[ie,:,1,iq] = ders[1,:]
            basis1[ie,:,2,iq] = ders[2,:]

    degree         = p6
    #..
    ne, nq         = points2.shape
    basis2         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = points2[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_6)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_6[low ]: 
                 span = low
            if xq >= knots_6[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_6[span] or xq >= knots_6[span+1]:
                 if xq < knots_6[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_6[span-j]
                right[j] = knots_6[span+1+j] - xq
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
            #~~~ Multiply derivatives by correct factors
            r = degree
            ders[1,:] = ders[1,:] * r
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis2[ie,:,0,iq] = ders[0,:]
            basis2[ie,:,1,iq] = ders[1,:]
            basis2[ie,:,2,iq] = ders[2,:]

    #--Computes coefficient of ratio in MAE = int rho_1/int rho_0
    C_ratio = 0.0
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]
    
            lcoeffs_z[ : , : ]  =  vector_z[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            duh_k = 0.0 
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol     = weights_3[ie1, g1]*weights_4[ie2, g2]

                    u_p        = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):
                          
                              coef_z = lcoeffs_z[il_1,il_2]
                              #...
                              bi_0   = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                              #...
                              u_p   += coef_z*bi_0

                    wvol     = weights_3[ie1, g1]*weights_4[ie2, g2]
                    C_ratio += wvol * (1.+intensity*u_p)

    # ...
    int_rhsP    = 0.0
    # ...
    lvalues_u1x = zeros((k1, k2))
    lvalues_u1y = zeros((k1, k2))
    lvalues_u2x = zeros((k1, k2))
    lvalues_u2y = zeros((k1, k2))
    rho         = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]


            # ...
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol       = weights_3[ie1, g1]*weights_4[ie2, g2]

                    #... We compute firstly the span in new adapted points
                    xq        = points1[ie2+ne2*ie1, g2+k2*g1]
                    degree    = p5
                    low       = degree
                    high      = len(knots_5)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_5[low ]: 
                         span = low
                    if xq >= knots_5[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_5[span] or xq >= knots_5[span+1]:
                         if xq < knots_5[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_5    = span
                    #...                    
                    xq        = points2[ie2+ne2*ie1, g2+k2*g1]
                    degree    = p6
                    low       = degree
                    high      = len(knots_6)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_6[low ]: 
                         span = low
                    if xq >= knots_6[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_6[span] or xq >= knots_6[span+1]:
                         if xq < knots_6[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_6    = span        

                    #------------------   
                    lcoeffs_z[ : , : ]  =  vector_z[span_5 : span_5+p5+1, span_6 : span_6+p6+1]
                    #------------------   
                    u_p        = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):

                              coef_z = lcoeffs_z[il_1,il_2]                                         
                              bi_0   = basis1[ie2+ne2*ie1,il_1,0,g2+k2*g1]*basis2[ie2+ne2*ie1,il_2,0,g2+k2*g1]
                              #...
                              u_p   += coef_z * bi_0
                              
                    rho[g1, g2] = C_ratio/ (1+intensity*u_p )


            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w[ : , : ]  = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            #...
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

                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2

            lvalues_u2x[ : , : ] = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        db1  = basis_4[ie1,il_1,1,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 

                            lvalues_u2x[g1,g2] += coeff_w*db1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2
            lvalues_u[ : , : ] = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    #...
                    u1x              = lvalues_u1x[g1,g2]
                    u1y              = lvalues_u1y[g1,g2]
                    #___
                    u2x              = lvalues_u2x[g1,g2]
                    u2y              = lvalues_u2y[g1,g2]
                    # ...
                    lvalues_u[g1,g2] = sqrt(u1x**2 + u2y**2 + 2. * rho[g1, g2] + 2.*u1y**2)
                    # ...
                    wvol             = weights_1[ie1, g1]*weights_2[ie2, g2]
                    int_rhsP        += sqrt(u1x**2 + u2y**2 + 2. * rho[g1, g2] + 2.*u1y**2)*wvol 
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p3+1):
                    i1 = i_span_4 - p4 + il_1
                    i2 = i_span_3 - p3 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_4[ie1, il_1, 0, g1] * basis_3[ie2, il_2, 0, g2]
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            #...
                            u     = lvalues_u[g1,g2]
                            #...
                            v    += bi_0 * u * wvol

                    rhs[i1+p4,i2+p3] += v   
    # Integral in Neumann boundary
    int_N = 2.
    # Assuring Compatiblity condition
    coefs = int_N/int_rhsP  
    rhs   = rhs*coefs
    # ...
    
#==============================================================================
#---Assemble rhs of Picard-Monge-Ampere equation
# with density function equal to sqrt(1.+ |grad(u_h)(F)|)
#==============================================================================
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int','int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'double[:,:]')
def assemble_vector_rhsmae(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, knots_1, knots_2, knots_3, knots_4, knots_5, knots_6, vector_u1, vector_w1, vector_z, left_v, right_v, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u1 = zeros((p1+1,p3+1))
    lcoeffs_w1 = zeros((p4+1,p2+1))
    lcoeffs_z  = zeros((p5+1,p6+1))
    
    lvalues_u  = zeros((k1, k2))
    Sol_weith  = zeros((ne1, ne2))
    
    points1    = zeros((ne1*ne2, k1*k2))
    points2    = zeros((ne1*ne2, k1*k2))

    # ... Assemble a new points by a new map 
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w1[ : , : ] = vector_w1[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              coeff_u = lcoeffs_u1[il_1,il_2]

                              sx     +=  coeff_u*bj_0
                    sy = 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              coeff_w = lcoeffs_w1[il_1,il_2]

                              sy     +=  coeff_w*bj_0
                    points1[ie2+ne2*ie1, g2+k2*g1] = sx
                    points2[ie2+ne2*ie1, g2+k2*g1] = sy

    #--Computes All basis in a new points
    nders          = 2
    quad_grid      = zeros((ne1*ne2, k1*k2))
    quad_grid[:,:] = points1[:,:]
    degree         = p5
    knots          = zeros(len(knots_5))
    knots[:]       = knots_5[:]
    #..
    ne, nq         = quad_grid.shape
    xx             = zeros(nq)

    left           = empty( degree )
    right          = empty( degree )
    ndu            = empty( (degree+1, degree+1) )
    a              = empty( (       2, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    basis1         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = quad_grid[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots[low ]: 
                 span = low
            if xq >= knots[high]: 
                 span = high-1
            else : 
              # Perform binary search
              span = (low+high)//2
              while xq < knots[span] or xq >= knots[span+1]:
                 if xq < knots[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots[span-j]
                right[j] = knots[span+1+j] - xq
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
            # Multiply derivatives by correct factors
            r = degree
            ders[1,:] = ders[1,:] * r
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis1[ie,:,0,iq] = ders[0,:]
            basis1[ie,:,1,iq] = ders[1,:]
            basis1[ie,:,2,iq] = ders[2,:]

    quad_grid[:,:] = points2[:,:]
    degree         = p6
    knots[:]       = knots_6[:]
    #..
    ne, nq         = quad_grid.shape
    basis2         = zeros( (ne,degree+1, nders+1, nq))
    for ie in range(ne):
        xx[:] = quad_grid[ie,:]
        for iq,xq in enumerate(xx):
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots[low ]: 
                 span = low
            if xq >= knots[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots[span] or xq >= knots[span+1]:
                 if xq < knots[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots[span-j]
                right[j] = knots[span+1+j] - xq
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
            #~~~ Multiply derivatives by correct factors
            r = degree
            ders[1,:] = ders[1,:] * r
            r = r * (degree-1)
            ders[2,:] = ders[2,:] * r
            basis2[ie,:,0,iq] = ders[0,:]
            basis2[ie,:,1,iq] = ders[1,:]
            basis2[ie,:,2,iq] = ders[2,:]
            
    #-- Computes coefficient d'intensity
    uhk_min = 0.0
    uhk_max = 0.0
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]
    
            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w1[ : , : ] = vector_w1[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_z[ : , : ]  =  vector_z[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            uhk = 0.0 
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol     = weights_3[ie1, g1]*weights_4[ie2, g2]

                    u_p        = 0.0
                    s_px       = 0.0
                    s_py       = 0.0
                    for il_1 in range(0, p5+1):
                        for il_2 in range(0, p6+1):

                              coef_z = lcoeffs_z[il_1,il_2]
                              bi_0   = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                              bi_x   = basis_5[ie1,il_1,1,g1]*basis_6[ie2,il_2,0,g2]
                              bi_y   = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,1,g2]
                              #...
                              u_p   += coef_z * bi_0
                              s_px  += coef_z * bi_x
                              s_py  += coef_z * bi_y
                              
                    lsxx    = 0.0
                    lsxy    = 0.0
                    lsyy    = 0.0                              
                    for il_3 in range(0, p1+1):
                        for il_4 in range(0, p3+1):
                              bj_x   = basis_1[ie1,il_3,1,g1]*basis_3[ie2,il_4,0,g2]
                              bj_y   = basis_1[ie1,il_3,0,g1]*basis_3[ie2,il_4,1,g2]
                              # ...
                              coef_u = lcoeffs_u1[il_3,il_4]
                              # ...
                              lsxx  += coef_u * bj_x
                              lsxy  += coef_u * bj_y
                    for il_3 in range(0, p4+1):
                        for il_4 in range(0, p2+1):
                              bj_y   = basis_4[ie1,il_3,0,g1]*basis_2[ie2,il_4,1,g2]
                              coef_w = lcoeffs_w1[il_3,il_4]
                              # ...
                              lsyy  += coef_w * bj_y

                    u_px  = (lsyy * s_px - lsxy * s_py)/(lsxx*lsyy-lsxy**2)
                    u_py  = (lsxx * s_py - lsxy * s_px)/(lsxx*lsyy-lsxy**2)

                    uhk   += (0.*u_p**2 + 1.*sqrt(u_px**2+ u_py**2))*wvol * (lsxx*lsyy-lsxy**2)
            Sol_weith[ie1,ie2] = uhk
            if uhk_max < uhk :
                 uhk_max = uhk  
            if uhk_min > uhk :
                 uhk_min = uhk
                 
    if   (uhk_max-uhk_min) <= 1e-4  :
         int_uh_0 = 0.
         int_uh_1 = 1.
    else :
        int_uh_0 = (right_v-left_v)/(uhk_max-uhk_min)
        int_uh_1  = (left_v*uhk_max-right_v*uhk_min)/(uhk_max-uhk_min)
    # ...
    J_mat      = zeros((k1,k2))
    #---
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]
    
            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w1[ : , : ] = vector_w1[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol     = weights_3[ie1, g1]*weights_4[ie2, g2]

                    lsxx    = 0.0
                    lsxy    = 0.0                           
                    for il_3 in range(0, p1+1):
                        for il_4 in range(0, p3+1):
                              bj_x   = basis_1[ie1,il_3,1,g1]*basis_3[ie2,il_4,0,g2]
                              bj_y   = basis_1[ie1,il_3,0,g1]*basis_3[ie2,il_4,1,g2]
                              coef_u = lcoeffs_u1[il_3,il_4]
                              # ...
                              lsxx  += coef_u * bj_x
                              lsxy  += coef_u * bj_y
                    lsyy    = 0.0   
                    for il_3 in range(0, p4+1):
                        for il_4 in range(0, p2+1):
                              bj_y   = basis_4[ie1,il_3,0,g1]*basis_2[ie2,il_4,1,g2]
                              coef_w = lcoeffs_w1[il_3,il_4]
                              # ...
                              lsyy  += coef_w * bj_y
                    # ...
                    J_mat[g1, g2] = (lsxx*lsyy-lsxy**2)
                    
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    for il_1 in range(0, p5+1):
                        for il_2 in range(0, p6+1):
                            xq     = points1[ie2+ne2*ie1, g2+k2*g1]

                            degree   = p5
                            knots[:] = knots_5[:]
                            low       = degree
                            high      = len(knots)-1-degree
                            # Check if point is exactly on left/right boundary, or outside domain
                            if xq <= knots[low ]: 
                                 span = low
                            if xq >= knots[high]: 
                                 span = high-1
                            else : 
                              # Perform binary search
                              span = (low+high)//2
                              while xq < knots[span] or xq >= knots[span+1]:
                                 if xq < knots[span]:
                                     high = span
                                 else:
                                     low  = span
                                 span = (low+high)//2

                            i1       = span - p5 + il_1
                            
                            # ...
                            xq       = points2[ie2+ne2*ie1, g2+k2*g1]

                            degree   = p6
                            knots[:] = knots_6[:]
                            low      = degree
                            high     = len(knots)-1-degree
                            # Check if point is exactly on left/right boundary, or outside domain
                            if xq <= knots[low ]: 
                                 span = low
                            if xq >= knots[high]: 
                                 span = high-1
                            else : 
                              # Perform binary search
                              span = (low+high)//2
                              while xq < knots[span] or xq >= knots[span+1]:
                                 if xq < knots[span]:
                                     high = span
                                 else:
                                     low  = span
                                 span = (low+high)//2

                            i2        = span - p6 + il_2
                            
                            # ...
                            bi_0      = basis1[ie2+ne2*ie1,il_1,0,g2+k2*g1]*basis2[ie2+ne2*ie1,il_2,0,g2+k2*g1]
                            # ...
                            wvol      = weights_5[ie1, g1]*weights_6[ie2, g2] 

                            #...
                            u     = (1.+abs(int_uh_0 * Sol_weith[ie1,ie2] + int_uh_1) )
                            #...
                            rhs[i1+p5,i2+p6] += bi_0 * u * wvol * J_mat[g1,g2]
    # ... 
