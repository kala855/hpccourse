# Ejemplo Básico

Este es un código de prueba que muestra el uso de MPI+CUDA, en este caso
se está utilizando un ejemplo muy simple en el cual se tienen 2 nodos,
el nodo maestro se va a encargar solo de enviar un número al nodo de
cálculo que tiene un GPU en la cual se calculará sólo el cuadrado del
número que se envía.

Desafortunadamente **Slurm** está fallando en la distribución de la
tarea, estoy tratando de solucionar el problema, sin embargo para
efectos de la entrega tendrán que usar **mpirun** directamente.

En la carpeta del código se encuentra un archivo **hostfile** que tiene
un listado de los equipos que estarán siendo usados como parte del
cluster.

Las lineas de compilación a usar pueden verificarse en el archivo compilation

Adicionalmente debe realizarse el siguiente proceso en cada una de las
cuentas:

## SSH Key

Se debe generar una llave RSA con el objetivo de garantizar la posibilidad de logearnos dentro del cluster sin el uso de password, esto le hace la vida mas fácil a **MPI**
Al generar la llave no usen passphrase.

```bash
ssh-keygen -t rsa
```

Esta llave debe adicionarse a las llaves autorizadas:

```bash
cd .ssh
cat id_rsa.pub >> authorized_keys
```

Adicionamos lo siguiente al `.bashrc` de nuestro usuario:

```bash
if type keychain >/dev/null 2>/dev/null; then
  keychain --nogui -q .ssh/id_rsa
  [ -f ~/.keychain/${HOSTNAME}-sh ] && . ~/.keychain/${HOSTNAME}-sh
  [ -f ~/.keychain/${HOSTNAME}-sh-gpg ] && . ~/.keychain/${HOSTNAME}-sh-gpg
fi
```

Nos deslogeamos e intentamos de nuevo. Lo que debe pasar es que al ejecutar por ejemplo:

```bash
ssh node01
```

desde cualquiera de los nodos del sistema deberíamos logearnos en el nodo sin solicitarnos password.

## Correr Código
Para ejecutar el código deben hacerlo de la siguiente manera:

```bash
mpirun -np 2 --hostfile hostfile mpi_cuda_hello_world
```
o para garantizar que se ejecuta en 2 nodos diferentes:

```bash
mpirun -host masterNode,node01 mpi_cuda_hello_world
```
