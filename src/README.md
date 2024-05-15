# Neurodegenerative Diffusion Model Design
Il design dell'applicazione prevede la suddivisione dal problema e dal solver. Questo permette di definire diversi 
problemi e parametrizzarli. 

- Classe NDProblem: contiene la definizione di un problema. 
- Classe NDSolver: contiene la definizione di un solver. Prende in pasto un NDProblem e lo risolve. In questo modo possiamo anche provare altri tipi diversi di solver.
- Classe NDSensitivity: seguendo quello che hanno fatto nel paper, avevo pensato di creare una sensitivity analysis per ogni parametro
del sistema e studiarne l'effetto sulla soluzione.

## NDProblem
**NDProblem** contiene i seguenti parametri:
1) Fiber Field -> il Diffusion Tensor direi che è costruito automaticamente.
2) extracellular diffusion
3) axonal diffusion
4) reaction coefficient
5) origin
6) delta t
7) T
8) Mesh
9) initial condition

## NDSolver
Il **NDSolver** contiene i seguenti parametri:
6) Preconditioner
7) Degree
9) Theta
11) Tolleranze CG e Newton
12) Max iterazioni CG e Newton


## NDAnalysis
Nella NDAnalysis si definisco le initial condition e il fiber field. Questi possono essere scelti da una gamma di 
IC e fiber field opportunamente definiti (radiale, circolare, radiale+circolare ecc...) che possono essere combinate per 
riprodurre i risultati del paper. Questi saranno configurabili attraverso dei parametri.

## TODO
- Ogni analisi dovrebbe essere salvata in un file in modo da poter essere riprodotta/identificare il problema appena eseguito.
- Ragionare sul precondizionatore.
- Ragionare su come affrontare la sensitivity analysis in modo non stupido.
- Ragionare su come impostare lo studio della scalabilità.
- Ragionare sulla linearizzazione del problema e altri metodi di risoluzione. (Taylor (?))

## Usage
```bash
./neuro_disease [-D dim] [-T T] [-a alpha] [-t deltat] [-g degree] [-e d_ext] [-x d_axn] [-m mesh] [-o output_filename] [-d output_dir]
```
Where:
- m: mesh file
- D: dimension of the problem
- a: growth factor
- e: extracellular diffusion coefficient
- x: axonal diffusion coefficient
- g: degree of the finite element
- t: time step
- T: final time
- o: output filename
- d: output directory


## Risorse
- Paper MOX sensitivity analysis: [exploring tau protein and amyloid-beta propagation: a sensitivity analysis of mathematical models based on biological data](https://mox.polimi.it/new-mox-report-on-exploring-tau-protein-and-amyloid-beta-propagation-a-sensitivity-analysis-of-mathematical-models-based-on-biological-data/)
- Initial condition: [ J. Weickenmeier, E. Kuhl, and A. Goriely, “Multiphysics of prionlike diseases: Progression and atrophy”,
Physical Review Letters, vol. 121, no. 15, p. 158101, 2018.](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.158101)