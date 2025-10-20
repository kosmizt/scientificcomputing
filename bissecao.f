      PROGRAM BISSECAO
      IMPLICIT NONE
      REAL A, B, XM, M, F, TOL
      INTEGER I      

      WRITE (*, *) "Entre com o valor de A: "
      READ (*, *) A
      WRITE (*, *) "Entre com o valor de B: "
      READ (*, *) B
      WRITE (*, *) "Entre com a tolerancia: "
      READ (*, *) TOL

 100  FORMAT (F8.5, 3X, F9.6)
 111  FORMAT (F9.6, 5X, F9.6)
 
      I = 0

      DO WHILE ((B-A) .GT. TOL)
         M = F(A)
         XM = (A+B)/2
         I = I + 1
         IF (M * F(XM) .GT. 0) THEN
           A = XM
         ELSE
           B = XM
         END IF
      END DO
      
      XM = (A+B)/2
      WRITE (*,*) "A RAIZ APROXIMADA:", XM
      WRITE (*,*) " EM ", I, " ITERACOES"
            
      END      
