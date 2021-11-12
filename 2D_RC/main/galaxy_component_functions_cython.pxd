

cimport cython

from typedefs cimport DTYPE_F32_t


cpdef DTYPE_F32_t vel_tot_iso(DTYPE_F32_t r, \
                              DTYPE_F32_t log_rhob0, \
                              DTYPE_F32_t Rb, \
                              DTYPE_F32_t SigD, \
                              DTYPE_F32_t Rd, \
                              DTYPE_F32_t rho0_h, \
                              DTYPE_F32_t Rh)

cpdef DTYPE_F32_t vel_tot_NFW(DTYPE_F32_t r, \
                              DTYPE_F32_t log_rhob0, \
                              DTYPE_F32_t Rb, \
                              DTYPE_F32_t SigD, \
                              DTYPE_F32_t Rd, \
                              DTYPE_F32_t rho0_h, \
                              DTYPE_F32_t Rh)

cpdef DTYPE_F32_t vel_tot_bur(DTYPE_F32_t r, \
                              DTYPE_F32_t log_rhob0, \
                              DTYPE_F32_t Rb, \
                              DTYPE_F32_t SigD, \
                              DTYPE_F32_t Rd, \
                              DTYPE_F32_t rho0_h, \
                              DTYPE_F32_t Rh)