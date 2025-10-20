      PROGRAM POSICAOF
      IMPLICIT NONE
      REAL A, B, XM, M, F, epsilonx, epsilony
      LOGICAL teste1, teste2, teste3
      INTEGER I
      
      WRITE (*, *) "Entre com o valor de A: "
      READ (*, *) A
      WRITE (*, *) "Entre com o valor de B: "
      READ (*, *) B
      WRITE (*, *) "Entre com a tolerancia no eixo x: "
      READ (*, *) epsilonx 
      WRITE (*, *) "Entre com a tolerancia no eixo y: "
      READ (*, *) epsilony 

 100  FORMAT (F8.5, 3X, F9.6)
 111  FORMAT (F9.6, 5X, F9.6)

      teste1 = ((B-A) .GT. epsilonx)
      teste2 = ((abs(F(A)) .GT. epsilony))
      teste3 = ((abs(F(B)) .GT. epsilony))
      
      I = 0
      
      DO WHILE (teste1 .AND. teste2 .AND. teste3)
         M = F(A)
         XM = (A*F(B)-B*F(A))/(F(B)-F(A))
         I = I + 1
         IF ((abs(F(XM)) .LT. epsilony)) THEN
            WRITE (*,*) "A RAIZ APROXIMADA EH:", XM
            WRITE (*,*) " EM ", I, " ITERACOES"
            STOP
         ENDIF
         IF (M * F(XM) .GT. 0) THEN
            A = XM
         ELSE
            B = XM
         ENDIF
         teste1 = ((B-A) .GT. epsilonx)
         teste2 = ((abs(F(A)) .GT. epsilony))
         teste3 = ((abs(F(B)) .GT. epsilony))         
      END DO
      
      IF (.NOT. teste1) THEN
         XM = (A + B)/2
      ENDIF
      IF (.NOT. teste2) THEN
         XM = A
      ENDIF
      IF (.NOT. teste3) THEN
         XM = B
      ENDIF
      
      WRITE (*,*) "A RAIZ APROXIMADA EH:", XM
      WRITE (*,*) " EM ", I, " ITERACOES"
      
      END      
      

