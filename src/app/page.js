'use client'
import React, { useState, useCallback } from 'react';
import { Calculator, Plus, Minus, Play, RotateCcw, BookOpen, Lightbulb, Settings } from 'lucide-react';
import { solveStandardMaximization, solveStandardMinimization } from './utils/simplexStandard'; 
import { solveBigMMaximization, solveBigMMinimization } from './utils/simplexBigM';


const SimplexSolver = () => {
  const [numVariables, setNumVariables] = useState(2);
  const [numConstraints, setNumConstraints] = useState(2);
  const [objective, setObjective] = useState([1, 1]);
  const [constraints, setConstraints] = useState([[1, 1], [2, 1]]);
  const [rhs, setRhs] = useState([4, 6]);
  const [isMaximization, setIsMaximization] = useState(true);
  const [inequalities, setInequalities] = useState(['<=', '<=']);
  const [method, setMethod] = useState('standard'); // 'standard', 'dual', 'bigM'
  const [solution, setSolution] = useState(null);
  const [steps, setSteps] = useState([]);
  const [showSteps, setShowSteps] = useState(false);

  // Classe principale pour résoudre le simplexe avec toutes les méthodes
  class SimplexSolver {
    constructor(c, A, b, inequalities, isMax = true, method = 'standard') {
      this.originalC = [...c];
      this.originalA = A.map(row => [...row]);
      this.originalB = [...b];
      this.originalInequalities = [...inequalities];
      this.isMax = isMax;
      this.method = method;
      this.numVars = c.length;
      this.numConstraints = b.length;
      this.steps = [];
      this.tableau = [];
      this.basicVars = [];
      this.nonBasicVars = [];
      this.M = 1000000; // Grande valeur pour la méthode du grand M
    }

    solve() {
      try {
        // Vérifier la faisabilité du problème
        if (!this.checkFeasibility()) {
          return { 
            status: 'infeasible', 
            message: 'Le problème n\'est pas réalisable',
            steps: this.steps 
          };
        }

        switch (this.method) {
          case 'dual':
            return this.solveDual();
          case 'bigM':
            return this.solveBigM();
          default:
            return this.solveStandard();
        }
      } catch (error) {
        return { 
          status: 'error', 
          message: error.message,
          steps: this.steps 
        };
      }
    }

    checkFeasibility() {
      // Vérifier si les contraintes sont cohérentes
      for (let i = 0; i < this.numConstraints; i++) {
        if (this.originalB[i] < 0 && this.originalInequalities[i] === '>=') {
          return false;
        }
      }
      return true;
    }

    // MÉTHODE STANDARD (améliorée)
    solveStandard() {
      this.steps.push({
        type: 'method',
        message: 'Utilisation de la méthode du simplexe standard'
      });

      // Vérifier si on a besoin de variables artificielles
      let needsArtificial = this.originalInequalities.some(ineq => ineq === '>=' || ineq === '=');
      
      if (needsArtificial) {
        this.steps.push({
          type: 'warning',
          message: 'Des contraintes d\'égalité ou ≥ détectées. Utilisation de la méthode du grand M recommandée.'
        });
        return this.solveBigM();
      }

      this.setupStandardTableau();
      return this.solveTableau();
    }

    setupStandardTableau() {
      // Conversion pour maximisation (si nécessaire)
      this.c = this.isMax ? this.originalC.map(x => -x) : [...this.originalC];
      this.A = this.originalA.map(row => [...row]);
      this.b = [...this.originalB];

      // Ajouter variables d'écart
      let slackVars = 0;
      for (let i = 0; i < this.numConstraints; i++) {
        if (this.originalInequalities[i] === '<=') {
          slackVars++;
        }
      }

      const totalVars = this.numVars + slackVars;
      const rows = this.numConstraints + 1;
      const cols = totalVars + 1;

      this.tableau = Array(rows).fill().map(() => Array(cols).fill(0));
      this.basicVars = [];
      this.nonBasicVars = [];

      // Variables non-basiques initiales
      for (let i = 0; i < this.numVars; i++) {
        this.nonBasicVars.push(i);
      }

      // Remplir les contraintes
      let slackIndex = this.numVars;
      for (let i = 0; i < this.numConstraints; i++) {
        // Variables originales
        for (let j = 0; j < this.numVars; j++) {
          this.tableau[i][j] = this.A[i][j];
        }

        // Variable d'écart
        if (this.originalInequalities[i] === '<=') {
          this.tableau[i][slackIndex] = 1;
          this.basicVars.push(slackIndex);
          slackIndex++;
        }

        // RHS
        this.tableau[i][cols - 1] = this.b[i];
      }

      // Fonction objectif
      for (let j = 0; j < this.numVars; j++) {
        this.tableau[rows - 1][j] = this.c[j];
      }

      this.steps.push({
        type: 'initial',
        message: 'Tableau initial (méthode standard)',
        tableau: this.copyTableau()
      });
    }

    // MÉTHODE DUALE
    solveDual() {
      this.steps.push({
        type: 'method',
        message: 'Utilisation de la méthode duale du simplexe'
      });

      // Construire le problème dual
      const dualProblem = this.constructDualProblem();
      
      this.steps.push({
        type: 'dual_construction',
        message: 'Construction du problème dual',
        dualProblem: dualProblem
      });

      // Résoudre le dual avec la méthode standard
      this.setupDualTableau(dualProblem);
      const dualResult = this.solveTableau();

      if (dualResult.status === 'optimal') {
        // Extraire la solution du primal à partir du dual
        const primalSolution = this.extractPrimalFromDual(dualResult.solution);
        
        this.steps.push({
          type: 'dual_to_primal',
          message: 'Conversion de la solution duale vers la solution primale',
          dualSolution: dualResult.solution,
          primalSolution: primalSolution
        });

        return {
          status: 'optimal',
          solution: primalSolution,
          dualSolution: dualResult.solution,
          steps: this.steps
        };
      }

      return dualResult;
    }

    constructDualProblem() {
      // Pour un problème primal de maximisation:
      // max c^T x subject to Ax <= b, x >= 0
      // Le dual est: min b^T y subject to A^T y >= c, y >= 0

      const dualObjective = [...this.originalB];
      const dualConstraints = [];
      const dualRHS = [...this.originalC];
      const dualInequalities = [];

      // Transposer la matrice A
      for (let j = 0; j < this.numVars; j++) {
        const row = [];
        for (let i = 0; i < this.numConstraints; i++) {
          row.push(this.originalA[i][j]);
        }
        dualConstraints.push(row);
        dualInequalities.push(this.isMax ? '>=' : '<=');
      }

      return {
        objective: dualObjective,
        constraints: dualConstraints,
        rhs: dualRHS,
        inequalities: dualInequalities,
        isMax: !this.isMax // Le dual d'un max est un min et vice versa
      };
    }

    setupDualTableau(dualProblem) {
      this.c = dualProblem.isMax ? dualProblem.objective.map(x => -x) : [...dualProblem.objective];
      this.A = dualProblem.constraints.map(row => [...row]);
      this.b = [...dualProblem.rhs];

      // Pour le dual, on a généralement besoin de variables d'excès et artificielles
      const needsArtificial = true; // Le dual a souvent des contraintes >=
      
      if (needsArtificial) {
        this.setupBigMTableau(this.c, this.A, this.b, dualProblem.inequalities);
      } else {
        this.setupStandardTableau();
      }
    }

    extractPrimalFromDual(dualSolution) {
      // La solution primale se trouve dans les coûts réduits du tableau final dual
      const primalVariables = [];
      
      // Pour chaque variable primale, regarder le coût réduit correspondant dans le dual
      const lastRow = this.tableau[this.tableau.length - 1];
      
      for (let i = 0; i < this.numVars; i++) {
        // La valeur de x_i est le coût réduit de la contrainte i dans le dual
        primalVariables.push(Math.abs(lastRow[this.numConstraints + i] || 0));
      }

      const objectiveValue = this.isMax ? 
        -lastRow[lastRow.length - 1] : lastRow[lastRow.length - 1];

      return {
        variables: primalVariables,
        objectiveValue: objectiveValue
      };
    }

    // MÉTHODE DU GRAND M
    solveBigM() {
      this.steps.push({
        type: 'method',
        message: 'Utilisation de la méthode du grand M'
      });

      this.setupBigMTableau(
        this.originalC, 
        this.originalA, 
        this.originalB, 
        this.originalInequalities
      );
      
      const result = this.solveTableau();
      
      if (result.status === 'optimal') {
        // Vérifier qu'aucune variable artificielle n'est dans la base
        if (this.hasArtificialInBasis()) {
          return {
            status: 'infeasible',
            message: 'Le problème n\'est pas réalisable (variables artificielles en base)',
            steps: this.steps
          };
        }
      }
      
      return result;
    }

    setupBigMTableau(c, A, b, inequalities) {
      // Convertir en forme de minimisation si nécessaire
      this.c = this.isMax ? c.map(x => -x) : [...c];
      this.A = A.map(row => [...row]);
      this.b = [...b];

      // Compter les variables nécessaires
      let numSlack = 0;
      let numSurplus = 0;
      let numArtificial = 0;

      inequalities.forEach(ineq => {
        if (ineq === '<=') numSlack++;
        else if (ineq === '>=') {
          numSurplus++;
          numArtificial++;
        } else if (ineq === '=') {
          numArtificial++;
        }
      });

      const totalVars = this.numVars + numSlack + numSurplus + numArtificial;
      const rows = this.numConstraints + 1;
      const cols = totalVars + 1;

      this.tableau = Array(rows).fill().map(() => Array(cols).fill(0));
      this.basicVars = [];
      this.nonBasicVars = [];
      this.artificialVars = [];

      // Variables non-basiques initiales
      for (let i = 0; i < this.numVars; i++) {
        this.nonBasicVars.push(i);
      }

      // Remplir les contraintes
      let slackIndex = this.numVars;
      let surplusIndex = this.numVars + numSlack;
      let artificialIndex = this.numVars + numSlack + numSurplus;

      for (let i = 0; i < this.numConstraints; i++) {
        // Variables originales
        for (let j = 0; j < this.numVars; j++) {
          this.tableau[i][j] = this.A[i][j];
        }

        // RHS
        this.tableau[i][cols - 1] = this.b[i];

        if (inequalities[i] === '<=') {
          // Variable d'écart
          this.tableau[i][slackIndex] = 1;
          this.basicVars.push(slackIndex);
          slackIndex++;
        } else if (inequalities[i] === '>=') {
          // Variable d'excès
          this.tableau[i][surplusIndex] = -1;
          this.nonBasicVars.push(surplusIndex);
          surplusIndex++;
          
          // Variable artificielle
          this.tableau[i][artificialIndex] = 1;
          this.basicVars.push(artificialIndex);
          this.artificialVars.push(artificialIndex);
          artificialIndex++;
        } else if (inequalities[i] === '=') {
          // Variable artificielle
          this.tableau[i][artificialIndex] = 1;
          this.basicVars.push(artificialIndex);
          this.artificialVars.push(artificialIndex);
          artificialIndex++;
        }
      }

      // Fonction objectif avec pénalités pour les variables artificielles
      for (let j = 0; j < this.numVars; j++) {
        this.tableau[rows - 1][j] = this.c[j];
      }

      // Ajouter les pénalités M pour les variables artificielles
      this.artificialVars.forEach(artVar => {
        this.tableau[rows - 1][artVar] = this.M;
      });

      // Éliminer les variables artificielles de la fonction objectif
      this.eliminateArtificialFromObjective();

      this.steps.push({
        type: 'initial',
        message: 'Tableau initial (méthode du grand M)',
        tableau: this.copyTableau(),
        artificialVars: [...this.artificialVars]
      });
    }

    eliminateArtificialFromObjective() {
      // Pour chaque variable artificielle en base, éliminer son coefficient de la fonction objectif
      const objRow = this.tableau.length - 1;
      
      this.artificialVars.forEach((artVar, index) => {
        if (this.basicVars.includes(artVar)) {
          const constraintRow = this.basicVars.indexOf(artVar);
          const multiplier = this.tableau[objRow][artVar];
          
          // Soustraire multiplier * ligne_contrainte de la fonction objectif
          for (let j = 0; j < this.tableau[objRow].length; j++) {
            this.tableau[objRow][j] -= multiplier * this.tableau[constraintRow][j];
          }
        }
      });
    }

    hasArtificialInBasis() {
      return this.artificialVars.some(artVar => 
        this.basicVars.includes(artVar) && 
        this.tableau[this.basicVars.indexOf(artVar)][this.tableau[0].length - 1] > 1e-10
      );
    }

    // ALGORITHME PRINCIPAL DU SIMPLEXE
    solveTableau() {
      let iteration = 0;
      const maxIterations = 100;

      while (!this.isOptimal() && iteration < maxIterations) {
        iteration++;

        const enteringVar = this.findEnteringVariable();
        if (enteringVar === -1) break;

        const leavingVarIndex = this.findLeavingVariable(enteringVar);
        if (leavingVarIndex === -1) {
          this.steps.push({
            type: 'unbounded',
            message: 'Le problème n\'a pas de solution bornée'
          });
          return { status: 'unbounded', steps: this.steps };
        }

        const leavingVar = this.basicVars[leavingVarIndex];
        this.pivot(leavingVarIndex, enteringVar);
        
        // Mise à jour des variables de base
        this.basicVars[leavingVarIndex] = enteringVar;
        
        // Mise à jour des variables non-basiques
        const enteringIndex = this.nonBasicVars.indexOf(enteringVar);
        if (enteringIndex !== -1) {
          this.nonBasicVars[enteringIndex] = leavingVar;
        }

        this.steps.push({
          type: 'iteration',
          message: `Itération ${iteration}: x${enteringVar + 1} entre, x${leavingVar + 1} sort`,
          tableau: this.copyTableau(),
          entering: enteringVar,
          leaving: leavingVar
        });
      }

      if (iteration >= maxIterations) {
        return { status: 'max_iterations', steps: this.steps };
      }

      const result = this.extractSolution();
      this.steps.push({
        type: 'optimal',
        message: 'Solution optimale trouvée',
        solution: result
      });

      return { status: 'optimal', solution: result, steps: this.steps };
    }

    isOptimal() {
      const lastRow = this.tableau[this.tableau.length - 1];
      for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < -1e-10) return false;
      }
      return true;
    }

    findEnteringVariable() {
      const lastRow = this.tableau[this.tableau.length - 1];
      let minValue = 0;
      let enteringVar = -1;

      for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < minValue) {
          minValue = lastRow[i];
          enteringVar = i;
        }
      }

      return enteringVar;
    }

    findLeavingVariable(enteringVar) {
      let minRatio = Infinity;
      let leavingVarIndex = -1;

      for (let i = 0; i < this.numConstraints; i++) {
        const pivot = this.tableau[i][enteringVar];
        if (pivot > 1e-10) {
          const ratio = this.tableau[i][this.tableau[0].length - 1] / pivot;
          if (ratio >= 0 && ratio < minRatio) {
            minRatio = ratio;
            leavingVarIndex = i;
          }
        }
      }

      return leavingVarIndex;
    }

    pivot(pivotRow, pivotCol) {
      const pivotElement = this.tableau[pivotRow][pivotCol];

      // Normaliser la ligne pivot
      for (let j = 0; j < this.tableau[pivotRow].length; j++) {
        this.tableau[pivotRow][j] /= pivotElement;
      }

      // Éliminer les autres éléments de la colonne pivot
      for (let i = 0; i < this.tableau.length; i++) {
        if (i !== pivotRow) {
          const factor = this.tableau[i][pivotCol];
          for (let j = 0; j < this.tableau[i].length; j++) {
            this.tableau[i][j] -= factor * this.tableau[pivotRow][j];
          }
        }
      }
    }

    extractSolution() {
      const solution = Array(this.numVars).fill(0);

      for (let i = 0; i < this.basicVars.length; i++) {
        const varIndex = this.basicVars[i];
        if (varIndex < this.numVars) {
          solution[varIndex] = this.tableau[i][this.tableau[0].length - 1];
        }
      }

      const objectiveValue = this.isMax ? 
        -this.tableau[this.tableau.length - 1][this.tableau[0].length - 1] :
        this.tableau[this.tableau.length - 1][this.tableau[0].length - 1];

      return {
        variables: solution,
        objectiveValue: objectiveValue
      };
    }

    copyTableau() {
      return this.tableau.map(row => [...row]);
    }
  }

  // Fonctions utilitaires pour l'interface
  const updateConstraint = (i, j, value) => {
    const newConstraints = [...constraints];
    newConstraints[i][j] = parseFloat(value) || 0;
    setConstraints(newConstraints);
  };

  const updateObjective = (i, value) => {
    const newObjective = [...objective];
    newObjective[i] = parseFloat(value) || 0;
    setObjective(newObjective);
  };

  const updateRHS = (i, value) => {
    const newRHS = [...rhs];
    newRHS[i] = parseFloat(value) || 0;
    setRhs(newRHS);
  };

  const updateInequality = (i, value) => {
    const newInequalities = [...inequalities];
    newInequalities[i] = value;
    setInequalities(newInequalities);
  };

  const addConstraint = () => {
    setNumConstraints(prev => {
      const newNum = prev + 1;
      setConstraints(prev => [...prev, Array(numVariables).fill(0)]);
      setRhs(prev => [...prev, 0]);
      setInequalities(prev => [...prev, '<=']);
      return newNum;
    });
  };

  const removeConstraint = () => {
    if (numConstraints > 1) {
      setNumConstraints(prev => prev - 1);
      setConstraints(prev => prev.slice(0, -1));
      setRhs(prev => prev.slice(0, -1));
      setInequalities(prev => prev.slice(0, -1));
    }
  };

  const addVariable = () => {
    setNumVariables(prev => {
      const newNum = prev + 1;
      setObjective(prev => [...prev, 0]);
      setConstraints(prev => prev.map(row => [...row, 0]));
      return newNum;
    });
  };

  const removeVariable = () => {
    if (numVariables > 2) {
      setNumVariables(prev => prev - 1);
      setObjective(prev => prev.slice(0, -1));
      setConstraints(prev => prev.map(row => row.slice(0, -1)));
    }
  };

  const solveProblem = () => {
    /*const solver = new SimplexSolver(objective, constraints, rhs, inequalities, isMaximization, method);
    const result = solver.solve();
    setSolution(result);
    setSteps(result.steps || []);*/
    let result;
    // Réinitialiser solution et étapes
    setSolution(null);
    setSteps([]);

    if (method === 'standard') {
        if (isMaximization) {
            result = solveStandardMaximization(objective, constraints, rhs, inequalities);
        } else { // Minimisation
            result = solveStandardMinimization(objective, constraints, rhs, inequalities);
        }
    } else if (method === 'dual') {
        // Logique pour le dual (à implémenter/séparer plus tard)
        // Pour l'instant, on peut rediriger vers Big M ou une placeholder
        const solver = new SimplexSolver(objective, constraints, rhs, inequalities, isMaximization, 'dual');
        result = solver.solve(); // Utilise l'ancienne classe pour l'instant
        console.warn("La méthode Duale est en cours de refactorisation. Utilisation de l'ancienne logique.");
    } else if (method === 'bigM') {
        // Logique pour Big M (à implémenter/séparer plus tard)
        if (isMaximization) {
            result = solveBigMMaximization(objective, constraints, rhs, inequalities);
        } else {
            result = solveBigMMinimization(objective, constraints, rhs, inequalities);
        }
    } else {
        result = { status: 'error', message: 'Méthode de résolution non reconnue.', steps: [] };
    }

    if (result) {
        setSolution(result); // result doit avoir la structure { status, solution?, objectiveValue?, steps }
        setSteps(result.steps || []); // Assurer que steps est toujours un tableau
        if (result.steps && result.steps.length > 0) {
            setShowSteps(true);
        }
    } else {
        // Gérer le cas où result est undefined, bien que cela ne devrait pas arriver avec la structure actuelle.
        setSolution({ status: 'error', message: 'Aucun résultat produit par le solveur.', steps: [] });
        setSteps([]);
    }
  };

  const resetProblem = () => {
    setSolution(null);
    setSteps([]);
    setShowSteps(false);
  };

  const formatNumber = (num) => {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    return Math.abs(num) < 1e-10 ? '0' : num.toFixed(4);
  };

  const getMethodDescription = () => {
    switch (method) {
      case 'standard':
        return 'Méthode standard pour problèmes avec contraintes ≤ uniquement';
      case 'dual':
        return 'Méthode duale - résout le problème dual puis extrait la solution primale';
      case 'bigM':
        return 'Méthode du grand M - gère tous types de contraintes (≤, ≥, =)';
      default:
        return '';
    }


    // app/page.js
// ... (imports existants)

// ...

  
    /*let result;
    setSolution(null);
    setSteps([]);

    if (method === 'standard') {
        if (isMaximization) {
            result = solveStandardMaximization(objective, constraints, rhs, inequalities);
        } else { 
            result = solveStandardMinimization(objective, constraints, rhs, inequalities);
        }
    } else if (method === 'bigM') {
        if (isMaximization) {
            result = solveBigMMaximization(objective, constraints, rhs, inequalities);
        } else {
            result = solveBigMMinimization(objective, constraints, rhs, inequalities);
        }
    } else if (method === 'dual') { 
        // ... (votre logique pour le dual explicite si vous le gardez)
        // Ou rediriger vers StandardMinimisation qui peut utiliser une approche duale
        if (!isMaximization) {
            steps.push({ type: 'info', message: 'Le bouton "Dual" appelle "Standard Minimisation" qui peut utiliser une approche duale.'});
            result = solveStandardMinimization(objective, constraints, rhs, inequalities); // StandardMin a la logique duale
        } else {
            result = { status: 'error', message: "Maximisation par Dual non typique. Utiliser Standard Max ou BigM Max.", steps:[]};
        }
    } else {
        result = { status: 'error', message: 'Méthode de résolution non reconnue.', steps: [] };
    }

    if (result) {
        setSolution(result);
        setSteps(result.steps || []);
        if (result.steps && result.steps.length > 0) {
            setShowSteps(true);
        }
    } else {
        setSolution({ status: 'error', message: 'Aucun résultat produit par le solveur.', steps: [] });
        setSteps([]);
    }*/
  
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl mb-6 shadow-2xl">
            <Calculator className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            Solveur Simplexe Complet
          </h1>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Toutes les méthodes : Standard, Duale et Grand M
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Configuration du problème */}
          <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div className="flex items-center gap-3 mb-8">
              <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center">
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-white">Configuration du Problème</h2>
            </div>

            {/* Sélection de la méthode */}
            <div className="mb-8">
              <label className="block text-lg font-semibold text-slate-200 mb-4">Méthode de résolution</label>
              <div className="space-y-3">
                <div className="flex gap-2">
                  <button
                    onClick={() => setMethod('standard')}
                    className={`flex-1 px-4 py-3 rounded-xl font-medium transition-all duration-300 ${method === 'standard'
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
                  >
                    Standard
                  </button>
                  <button
                    onClick={() => setMethod('dual')}
                    className={`flex-1 px-4 py-3 rounded-xl font-medium transition-all duration-300 ${method === 'dual'
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
                  >
                    Duale
                  </button>
                  <button
                    onClick={() => setMethod('bigM')}
                    className={`flex-1 px-4 py-3 rounded-xl font-medium transition-all duration-300 ${method === 'bigM'
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
                  >
                    Grand M
                  </button>
                </div>
                <p className="text-sm text-slate-400 bg-white/5 rounded-lg p-3">
                  {getMethodDescription()}
                </p>
              </div>
            </div>

            {/* Type d'optimisation */}
            <div className="mb-8">
              <label className="block text-lg font-semibold text-slate-200 mb-4">Type d'optimisation</label>
              <div className="flex gap-4">
                <button
                  onClick={() => setIsMaximization(true)}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${isMaximization
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                      : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
                >
                  Maximisation
                </button>
                <button
                  onClick={() => setIsMaximization(false)}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${!isMaximization
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                      : 'bg-white/10 text-slate-300 hover:bg-white/20'}`}
                >
                  Minimisation
                </button>
              </div>
            </div>

            {/* Fonction objectif */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <label className="text-lg font-semibold text-slate-200">
                  Fonction objectif: {isMaximization ? 'Maximiser' : 'Minimiser'} Z =
                </label>
                <div className="flex gap-2">
                  <button
                    onClick={addVariable}
                    className="w-8 h-8 bg-green-500/20 hover:bg-green-500/30 rounded-lg flex items-center justify-center transition-colors"
                  >
                    <Plus className="w-4 h-4 text-green-400" />
                  </button>
                  <button
                    onClick={removeVariable}
                    disabled={numVariables <= 2}
                    className="w-8 h-8 bg-red-500/20 hover:bg-red-500/30 rounded-lg flex items-center justify-center transition-colors disabled:opacity-50"
                  >
                    <Minus className="w-4 h-4 text-red-400" />
                  </button>
                </div>
              </div>
              <div className="flex flex-wrap gap-3 items-center">
                {objective.map((coef, i) => (
                  <div key={i} className="flex items-center gap-2">
                    {i > 0 && <span className="text-slate-300 text-lg">+</span>}
                    <input
                      type="number"
                      value={coef}
                      onChange={(e) => updateObjective(i, e.target.value)}
                      className="w-16 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-center focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      step="0.1" />
                    <span className="text-slate-300 font-medium">x{i + 1}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Contraintes */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <label className="text-lg font-semibold text-slate-200">Contraintes</label>
                <div className="flex gap-2">
                  <button
                    onClick={addConstraint}
                    className="w-8 h-8 bg-green-500/20 hover:bg-green-500/30 rounded-lg flex items-center justify-center transition-colors"
                  >
                    <Plus className="w-4 h-4 text-green-400" />
                  </button>
                  <button
                    onClick={removeConstraint}
                    disabled={numConstraints <= 1}
                    className="w-8 h-8 bg-red-500/20 hover:bg-red-500/30 rounded-lg flex items-center justify-center transition-colors disabled:opacity-50"
                  >
                    <Minus className="w-4 h-4 text-red-400" />
                  </button>
                </div>
              </div>
              <div className="space-y-4">
                {constraints.map((constraint, i) => (
                  <div key={i} className="flex flex-wrap gap-3 items-center p-4 bg-white/5 rounded-xl border border-white/10">
                    {constraint.map((coef, j) => (
                      <div key={j} className="flex items-center gap-2">
                        {j > 0 && <span className="text-slate-300">+</span>}
                        <input
                          type="number"
                          value={coef}
                          onChange={(e) => updateConstraint(i, j, e.target.value)}
                          className="w-16 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-center focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          step="0.1" />
                        <span className="text-slate-300 font-medium">x{j + 1}</span>
                      </div>
                    ))}
                    <select
                      value={inequalities[i]}
                      onChange={(e) => updateInequality(i, e.target.value)}
                      className="px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="<=">≤</option>
                      <option value="=">=</option>
                      <option value=">=">≥</option>
                    </select>
                    <input
                      type="number"
                      value={rhs[i]}
                      onChange={(e) => updateRHS(i, e.target.value)}
                      className="w-20 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-center focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      step="0.1" />
                  </div>
                ))}
              </div>
            </div>

            {/* Boutons d'action */}
            <div className="flex gap-4">
              <button
                onClick={solveProblem}
                className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 px-6 py-4 rounded-xl text-white font-semibold transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center gap-3"
              >
                <Play className="w-5 h-5" />
                Résoudre
              </button>
              <button
                onClick={resetProblem}
                className="px-6 py-4 bg-white/10 hover:bg-white/20 rounded-xl text-white font-semibold transition-all duration-300 border border-white/20 flex items-center justify-center gap-3"
              >
                <RotateCcw className="w-5 h-5" />
                Reset
              </button>
            </div>
          </div>

          {/* Résultats */}
          <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-8 border border-white/20 shadow-2xl">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">Résultats</h2>
              </div>
              {steps.length > 0 && (
                <button
                  onClick={() => setShowSteps(!showSteps)}
                  className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white font-medium transition-all duration-300 border border-white/20"
                >
                  {showSteps ? 'Masquer' : 'Voir'} les étapes
                </button>
              )}
            </div>

            {!solution ? (
              <div className="text-center py-12">
                <div className="w-24 h-24 bg-gradient-to-r from-slate-600 to-slate-700 rounded-full flex items-center justify-center mx-auto mb-6 opacity-50">
                  <Calculator className="w-12 h-12 text-slate-400" />
                </div>
                <p className="text-slate-400 text-lg">
                  Configurez votre problème et cliquez sur "Résoudre"
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {solution.status === 'optimal' && solution.solution && (
                  <div className="space-y-4">
                    <div className="bg-green-500/20 border border-green-500/30 rounded-xl p-6">
                      <h3 className="text-xl font-bold text-green-400 mb-4">Solution Optimale</h3>
                      <div className="space-y-3">
                        <div className="text-2xl font-bold text-white">
                          Z = {formatNumber(solution.solution.objectiveValue)}
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          {solution.solution.variables.map((value, i) => (
                            <div key={i} className="bg-white/10 rounded-lg p-3">
                              <span className="text-slate-300">x{i + 1} = </span>
                              <span className="text-white font-semibold">{formatNumber(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Afficher la solution duale si elle existe */}
                      {solution.dualSolution && (
                        <div className="mt-4 pt-4 border-t border-green-500/30">
                          <h4 className="text-lg font-semibold text-green-300 mb-2">Solution Duale</h4>
                          <div className="text-lg font-bold text-white mb-2">
                            Z_dual = {formatNumber(solution.dualSolution.objectiveValue)}
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            {solution.dualSolution.variables.map((value, i) => (
                              <div key={i} className="bg-white/5 rounded p-2 text-sm">
                                <span className="text-slate-300">y{i + 1} = </span>
                                <span className="text-white">{formatNumber(value)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {solution.status === 'unbounded' && (
                  <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-yellow-400 mb-2">Problème Non Borné</h3>
                    <p className="text-slate-300">Le problème n'a pas de solution optimale finie. La fonction objectif peut être améliorée indéfiniment.</p>
                  </div>
                )}

                {solution.status === 'infeasible' && (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-red-400 mb-2">Problème Non Réalisable</h3>
                    <p className="text-slate-300">{solution.message || "Les contraintes sont incompatibles. Il n'existe aucune solution satisfaisant toutes les contraintes."}</p>
                  </div>
                )}

                {solution.status === 'error' && (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-red-400 mb-2">Erreur</h3>
                    <p className="text-slate-300">{solution.message}</p>
                  </div>
                )}

                {solution.status === 'max_iterations' && (
                  <div className="bg-orange-500/20 border border-orange-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-orange-400 mb-2">Limite d'Itérations Atteinte</h3>
                    <p className="text-slate-300">L'algorithme n'a pas convergé dans le nombre maximum d'itérations autorisées.</p>
                  </div>
                )}

                {/* Étapes de résolution */}
                {showSteps && steps.length > 0 && (
                  <div className="space-y-4">
                    <h3 className="text-xl font-bold text-white">Étapes de Résolution</h3>
                    <div className="space-y-4 max-h-96 overflow-y-auto">
                      {steps.map((step, i) => (
                        <div key={i} className="bg-white/5 rounded-xl p-4 border border-white/10">
                          <div className="text-blue-400 font-semibold mb-2">
                            {step.type === 'method' && <span className="bg-blue-500/20 px-2 py-1 rounded text-xs mr-2">MÉTHODE</span>}
                            {step.type === 'initial' && <span className="bg-green-500/20 px-2 py-1 rounded text-xs mr-2">INITIAL</span>}
                            {step.type === 'iteration' && <span className="bg-purple-500/20 px-2 py-1 rounded text-xs mr-2">ITÉRATION</span>}
                            {step.type === 'optimal' && <span className="bg-emerald-500/20 px-2 py-1 rounded text-xs mr-2">OPTIMAL</span>}
                            {step.type === 'dual_construction' && <span className="bg-cyan-500/20 px-2 py-1 rounded text-xs mr-2">DUAL</span>}
                            {step.type === 'warning' && <span className="bg-yellow-500/20 px-2 py-1 rounded text-xs mr-2">ATTENTION</span>}
                            {step.message}
                          </div>
                          
                          {step.tableau && (
                            <div className="overflow-x-auto mt-3">
                              <table className="w-full text-xs border-collapse">
                                <tbody>
                                  {step.tableau.map((row, ri) => (
                                    <tr key={ri} className={ri === step.tableau.length - 1 ? 'border-t border-white/20' : ''}>
                                      {row.map((cell, ci) => (
                                        <td key={ci} className="px-2 py-1 text-center text-slate-300 border border-white/10">
                                          {formatNumber(cell)}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}

                          {step.dualProblem && (
                            <div className="mt-3 p-3 bg-white/5 rounded-lg">
                              <h5 className="text-sm font-semibold text-cyan-300 mb-2">Problème Dual Construit:</h5>
                              <div className="text-xs text-slate-300 space-y-1">
                                <div>Objectif: {step.dualProblem.isMax ? 'Max' : 'Min'} {step.dualProblem.objective.map(formatNumber).join(' + ')}</div>
                                <div>Contraintes: {step.dualProblem.constraints.length} contraintes</div>
                              </div>
                            </div>
                          )}

                          {step.artificialVars && step.artificialVars.length > 0 && (
                            <div className="mt-3 p-3 bg-red-500/10 rounded-lg">
                              <h5 className="text-sm font-semibold text-red-300 mb-1">Variables Artificielles:</h5>
                              <div className="text-xs text-slate-300">
                                Variables {step.artificialVars.map(v => `x${v+1}`).join(', ')} avec pénalité M
                              </div>
                            </div>
                          )}

                          {step.entering !== undefined && step.leaving !== undefined && (
                            <div className="mt-3 flex gap-4 text-xs">
                              <span className="text-green-300">Entre: x{step.entering + 1}</span>
                              <span className="text-red-300">Sort: x{step.leaving + 1}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Section éducative améliorée */}
        <div className="mt-16 bg-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/10 shadow-2xl">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">
            Les Trois Méthodes du Simplexe
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">S</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Méthode Standard</h3>
              <p className="text-slate-300 leading-relaxed text-sm">
                Utilisée pour les problèmes avec contraintes ≤ uniquement. Ajoute des variables d'écart pour transformer les inégalités en égalités. Méthode la plus simple et efficace pour ce type de problèmes.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">D</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Méthode Duale</h3>
              <p className="text-slate-300 leading-relaxed text-sm">
                Résout le problème dual au lieu du primal. Utile quand le dual a moins de contraintes que le primal, ou pour certains types d'analyses de sensibilité. La solution optimale est identique.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">M</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Méthode du Grand M</h3>
              <p className="text-slate-300 leading-relaxed text-sm">
                Méthode universelle qui gère tous types de contraintes (≤, ≥, =). Utilise des variables artificielles avec une pénalité M très grande. Indispensable pour les problèmes mixtes.
              </p>
            </div>
          </div>
          
          <div className="mt-8 p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl border border-blue-500/20">
            <h3 className="text-xl font-bold text-white mb-4 text-center">Guide de Sélection</h3>
            <div className="grid md:grid-cols-3 gap-6 text-sm">
              <div className="text-center">
                <div className="text-blue-300 font-semibold mb-2">Contraintes ≤ seulement</div>
                <div className="text-slate-300">→ Méthode Standard</div>
              </div>
              <div className="text-center">
                <div className="text-purple-300 font-semibold mb-2">Problème avec peu de contraintes</div>
                <div className="text-slate-300">→ Méthode Duale</div>
              </div>
              <div className="text-center">
                <div className="text-green-300 font-semibold mb-2">Contraintes mixtes (≤, ≥, =)</div>
                <div className="text-slate-300">→ Méthode du Grand M</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

          )
        }

export default SimplexSolver;