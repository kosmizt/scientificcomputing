      PROGRAM WIENLAW
      
      REAL*8 B, T, LAMBDA
      PARAMETER (B = 2.897771955D-3)

      PRINT *, 'Enter temperature in Kelvin:'
      READ *, T

      IF (T .LE. 0.0D0) THEN
         PRINT *, 'Temperature must be positive.'
         STOP
      END IF

      LAMBDA = B / T

      PRINT *, 'Peak wavelength (m):', LAMBDA

      END
