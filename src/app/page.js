'use client'
import React, { useState, useCallback } from 'react';
import { Calculator, Plus, Minus, Play, RotateCcw, BookOpen, Lightbulb } from 'lucide-react';

const SimplexSolver = () => {
  const [numVariables, setNumVariables] = useState(2);
  const [numConstraints, setNumConstraints] = useState(2);
  const [objective, setObjective] = useState([1, 1]);
  const [constraints, setConstraints] = useState([[1, 1], [2, 1]]);
  const [rhs, setRhs] = useState([4, 6]);
  const [isMaximization, setIsMaximization] = useState(true);
  const [inequalities, setInequalities] = useState(['<=', '<=']); // Nouvel état pour les inégalités
  const [solution, setSolution] = useState(null);
  const [steps, setSteps] = useState([]);
  const [showSteps, setShowSteps] = useState(false);

  // Classe pour résoudre le simplexe
  class SimplexSolver {
    constructor(c, A, b, inequalities, isMax = true) {
      this.c = isMax ? c.map(x => -x) : [...c]; // Convertir en minimisation
      this.A = A.map(row => [...row]);
      this.b = [...b];
      this.inequalities = [...inequalities];
      this.isMax = isMax;
      this.numVars = c.length;
      this.numConstraints = b.length;
      this.steps = [];
      this.tableau = [];
      this.basicVars = [];
      this.M = 1000; // Valeur du grand M
    }

    solve() {
      try {
        this.setupInitialTableau();
        this.steps.push({
          type: 'initial',
          message: 'Tableau initial du simplexe',
          tableau: this.copyTableau()
        });

        let iteration = 0;
        const maxIterations = 50;

        while (!this.isOptimal() && iteration < maxIterations) {
          iteration++;
          
          const enteringVar = this.findEnteringVariable();
          if (enteringVar === -1) break;

          const leavingVar = this.findLeavingVariable(enteringVar);
          if (leavingVar === -1) {
            this.steps.push({
              type: 'unbounded',
              message: 'Le problème n\'a pas de solution bornée (non borné)'
            });
            return { status: 'unbounded', steps: this.steps };
          }

          this.pivot(leavingVar, enteringVar);
          this.basicVars[leavingVar] = enteringVar;

          this.steps.push({
            type: 'iteration',
            message: `Itération ${iteration}: Variable x${enteringVar + 1} entre, variable de base ${leavingVar + 1} sort`,
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
      } catch (error) {
        return { 
          status: 'error', 
          message: error.message,
          steps: this.steps 
        };
      }
    }

    setupInitialTableau() {
      // Compter le nombre de variables d'écart, d'excès et artificielles nécessaires
      let numSlack = 0;
      let numSurplus = 0;
      let numArtificial = 0;
      
      this.inequalities.forEach(ineq => {
        if (ineq === '<=') numSlack++;
        else if (ineq === '>=') {
          numSurplus++;
          numArtificial++;
        }
        else if (ineq === '=') numArtificial++;
      });
      
      const totalNewVars = numSlack + numSurplus + numArtificial;
      const rows = this.numConstraints + 1;
      const cols = this.numVars + totalNewVars + 1;
      
      this.tableau = Array(rows).fill().map(() => Array(cols).fill(0));
      this.basicVars = [];
      
      // Variables artificielles pour la méthode du grand M
      const artificialVars = [];
      
      // Remplir les contraintes
      let slackIndex = 0;
      let artificialIndex = 0;
      
      for (let i = 0; i < this.numConstraints; i++) {
        // Variables originales
        for (let j = 0; j < this.numVars; j++) {
          this.tableau[i][j] = this.A[i][j];
        }
        
        const rhsCol = cols - 1;
        this.tableau[i][rhsCol] = this.b[i];
        
        // Gestion des inégalités
        if (this.inequalities[i] === '<=') {
          // Variable d'écart
          const slackPos = this.numVars + slackIndex;
          this.tableau[i][slackPos] = 1;
          this.basicVars.push(slackPos);
          slackIndex++;
        } 
        else if (this.inequalities[i] === '>=') {
          // Variable d'excès
          const surplusPos = this.numVars + slackIndex + numSurplus - 1;
          this.tableau[i][surplusPos] = -1;
          
          // Variable artificielle
          const artificialPos = this.numVars + numSlack + numSurplus + artificialIndex;
          this.tableau[i][artificialPos] = 1;
          artificialVars.push(artificialPos);
          this.basicVars.push(artificialPos);
          artificialIndex++;
        } 
        else if (this.inequalities[i] === '=') {
          // Variable artificielle
          const artificialPos = this.numVars + numSlack + numSurplus + artificialIndex;
          this.tableau[i][artificialPos] = 1;
          artificialVars.push(artificialPos);
          this.basicVars.push(artificialPos);
          artificialIndex++;
        }
      }
      
      // Fonction objectif originale
      for (let j = 0; j < this.numVars; j++) {
        this.tableau[rows - 1][j] = this.c[j];
      }
      
      // Fonction objectif pour les variables artificielles (méthode du grand M)
      if (artificialVars.length > 0) {
        // Créer une ligne temporaire pour la fonction objectif avec M
        const tempObjRow = Array(cols).fill(0);
        
        // Pour chaque variable artificielle, ajouter -M dans la fonction objectif
        artificialVars.forEach(artVar => {
          tempObjRow[artVar] = this.isMax ? -this.M : this.M;
        });
        
        // Soustraire cette ligne de la fonction objectif originale
        for (let j = 0; j < cols; j++) {
          this.tableau[rows - 1][j] += tempObjRow[j];
        }
      }
    }

    isOptimal() {
      const lastRow = this.tableau[this.tableau.length - 1];
      for (let i = 0; i < lastRow.length - 1; i++) {
        if (lastRow[i] < 0) return false;
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
      let leavingVar = -1;
      
      for (let i = 0; i < this.numConstraints; i++) {
        const pivot = this.tableau[i][enteringVar];
        if (pivot > 0) {
          const ratio = this.tableau[i][this.tableau[0].length - 1] / pivot;
          if (ratio < minRatio) {
            minRatio = ratio;
            leavingVar = i;
          }
        }
      }
      
      return leavingVar;
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
    const solver = new SimplexSolver(objective, constraints, rhs, inequalities, isMaximization);
    const result = solver.solve();
    setSolution(result);
    setSteps(result.steps || []);
  };

  const resetProblem = () => {
    setSolution(null);
    setSteps([]);
    setShowSteps(false);
  };

  const formatNumber = (num) => {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    return Math.abs(num) < 1e-10 ? '0' : num.toFixed(3);
  };

  return (
    <>
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="></div>

      {/*>60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%239C92AC" fill-opacity="0.1"%3E%3Cpath d="m36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-20" */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl mb-6 shadow-2xl">
            <Calculator className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
            Solveur Simplexe
          </h1>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Résolvez vos problèmes de programmation linéaire avec la méthode du simplexe
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
                    </div>
                  </div>
                )}

                {solution.status === 'unbounded' && (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-red-400 mb-2">Problème Non Borné</h3>
                    <p className="text-slate-300">Le problème n'a pas de solution optimale finie.</p>
                  </div>
                )}

                {solution.status === 'error' && (
                  <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6">
                    <h3 className="text-xl font-bold text-red-400 mb-2">Erreur</h3>
                    <p className="text-slate-300">{solution.message}</p>
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
                            Étape {i + 1}: {step.message}
                          </div>
                          {step.tableau && (
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs">
                                <tbody>
                                  {step.tableau.map((row, ri) => (
                                    <tr key={ri} className={ri === step.tableau.length - 1 ? 'border-t border-white/20' : ''}>
                                      {row.map((cell, ci) => (
                                        <td key={ci} className="px-2 py-1 text-center text-slate-300">
                                          {formatNumber(cell)}
                                        </td>
                                      ))}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
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

        {/* Section éducative */}
        <div className="mt-16 bg-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/10 shadow-2xl">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">
            Comment fonctionne la méthode du Simplexe ?
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">1</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Forme Standard</h3>
              <p className="text-slate-300 leading-relaxed">
                Conversion du problème en forme standard en ajoutant des variables d'écart, d'excès et artificielles pour transformer les inégalités en égalités.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">2</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Tableau Initial</h3>
              <p className="text-slate-300 leading-relaxed">
                Construction du tableau du simplexe avec les coefficients des contraintes et de la fonction objectif, en utilisant la méthode du grand M si nécessaire.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-white">3</span>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Itérations</h3>
              <p className="text-slate-300 leading-relaxed">
                Opérations de pivot successives jusqu'à atteindre la solution optimale ou détecter un problème non borné.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
    </>
  );
};

export default SimplexSolver;