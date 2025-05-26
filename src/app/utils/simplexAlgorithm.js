// app/utils/simplexAlgorithm.js

export class SimplexSolver {
  constructor(c, A, b, inequalities, optimizationType = 'maximize', method = 'primal_big_m') {
    this.original_c = [...c];
    this.original_A = A.map(row => [...row]);
    this.original_b = [...b];
    this.original_inequalities = [...inequalities]; // Celles fournies initialement par l'UI
    
    this.optimizationType = optimizationType; // 'maximize' ou 'minimize'
    this.method = method; // 'primal_big_m' ou 'dual'

    this.isMax = (this.optimizationType === 'maximize');

    // Ces `c, A, b, inequalities` seront celles utilisées pour construire le tableau,
    // potentiellement modifiées par les méthodes de préparation.
    this.c = []; 
    this.A = [];
    this.b = [];
    this.inequalities = []; // Celles utilisées pour le setup du tableau, ex: toutes '<=' pour le dual

    this.numVars = 0; // Nombre de variables de décision originales
    this.numConstraints = 0;
    this.originalVarCount = this.original_c.length; // Conserver pour affichage/étapes
    this.numSlackVars = 0;
    this.numSurplusVars = 0;
    this.numArtificialVars = 0; // Utilisé par primal_big_m

    this.steps = [];
    this.tableau = [];
    this.basicVars = []; // Indices des variables de base (par rapport aux colonnes du tableau)
    this.M = 1000; // Valeur du grand M pour primal
    this.epsilon = 1e-9; // Tolérance pour les comparaisons flottantes
  }

  solve() {
    this.steps = []; // Réinitialiser les étapes pour chaque solve
    try {
      if (this.method === 'dual') {
        return this._solveDual();
      } else { // primal_big_m (par défaut)
        return this._solvePrimalBigM();
      }
    } catch (error) {
      console.error(`Erreur dans SimplexSolver (${this.method}):`, error);
      this.steps.push({ type: 'error', message: `Erreur d'exécution: ${error.message || 'Erreur inconnue'}` });
      return { 
        status: 'error', 
        message: `Erreur d'exécution: ${error.message || 'Erreur inconnue'}`,
        steps: this.steps 
      };
    }
  }

  // --- Méthodes pour le Simplexe Primal avec Big M ---
  _prepareForPrimalBigM() {
    // Utilise this.original_c, original_A, original_b, original_inequalities
    // et initialise this.c, this.A, this.b, this.inequalities pour le primal
    this.c = this.isMax ? this.original_c.map(x => -x) : [...this.original_c];
    this.A = this.original_A.map(row => [...row]);
    this.b = [...this.original_b];
    this.inequalities = [...this.original_inequalities]; // Celles d'origine de l'UI
    
    this.numVars = this.c.length;
    this.numConstraints = this.b.length;

    // Comptage des variables à ajouter (slack, surplus, artificielles)
    this.numSlackVars = 0;
    this.numSurplusVars = 0;
    this.numArtificialVars = 0;

    for (let i = 0; i < this.numConstraints; i++) {
        if (this.b[i] < 0) { // Standardiser RHS pour qu'il soit positif
            this.A[i] = this.A[i].map(val => -val);
            this.b[i] = -this.b[i];
            if (this.inequalities[i] === '<=') this.inequalities[i] = '>=';
            else if (this.inequalities[i] === '>=') this.inequalities[i] = '<=';
        }

        if (this.inequalities[i] === '<=') this.numSlackVars++;
        else if (this.inequalities[i] === '>=') {
            this.numSurplusVars++;
            this.numArtificialVars++;
        } else if (this.inequalities[i] === '=') {
            this.numArtificialVars++;
        }
    }
  }

  _buildTableauForPrimalBigM() {
    const totalTableauVars = this.numVars + this.numSlackVars + this.numSurplusVars + this.numArtificialVars;
    const rows = this.numConstraints + 1;
    const cols = totalTableauVars + 1;
    
    this.tableau = Array(rows).fill(null).map(() => Array(cols).fill(0));
    this.basicVars = Array(this.numConstraints).fill(-1);
    
    let slackIdx = 0, surplusIdx = 0, artificialIdx = 0;

    for (let i = 0; i < this.numConstraints; i++) {
      for (let j = 0; j < this.numVars; j++) {
        this.tableau[i][j] = this.A[i][j];
      }
      this.tableau[i][cols - 1] = this.b[i];

      if (this.inequalities[i] === '<=') {
        const slackVarCol = this.numVars + slackIdx;
        this.tableau[i][slackVarCol] = 1;
        this.basicVars[i] = slackVarCol;
        slackIdx++;
      } else if (this.inequalities[i] === '>=') {
        const surplusVarCol = this.numVars + this.numSlackVars + surplusIdx;
        this.tableau[i][surplusVarCol] = -1;
        surplusIdx++;
        const artificialVarCol = this.numVars + this.numSlackVars + this.numSurplusVars + artificialIdx;
        this.tableau[i][artificialVarCol] = 1;
        this.basicVars[i] = artificialVarCol;
        artificialIdx++;
      } else if (this.inequalities[i] === '=') {
        const artificialVarCol = this.numVars + this.numSlackVars + this.numSurplusVars + artificialIdx;
        this.tableau[i][artificialVarCol] = 1;
        this.basicVars[i] = artificialVarCol;
        artificialIdx++;
      }
    }

    // Ligne Z initiale (avant ajustement Big M)
    for (let j = 0; j < this.numVars; j++) {
      this.tableau[rows - 1][j] = this.c[j];
    }

    // Ajustement Big M pour la ligne Z
    for (let i = 0; i < this.numConstraints; i++) {
      const basicVarForConstraint = this.basicVars[i];
      if (basicVarForConstraint >= this.numVars + this.numSlackVars + this.numSurplusVars) { // C'est une variable artificielle
        const M_value = this.M; // this.c est déjà ajusté (-c pour max), donc M est "positif"
        this.tableau[rows - 1][basicVarForConstraint] = M_value; // Coût M pour variable artificielle
        // Éliminer M de la ligne Z pour les variables artificielles de base
        for (let k = 0; k < cols; k++) {
          this.tableau[rows - 1][k] -= M_value * this.tableau[i][k];
        }
      }
    }
  }
  
  _solvePrimalBigM() {
    this._prepareForPrimalBigM();
    this._buildTableauForPrimalBigM();

    this.steps.push({
      type: 'initial', message: 'Tableau initial (Primal Big M)',
      tableau: this.copyTableau(), basicVars: [...this.basicVars],
      numOriginalVars: this.originalVarCount, numSlackVars: this.numSlackVars,
      numSurplusVars: this.numSurplusVars, numArtificialVars: this.numArtificialVars,
      varTypes: this._getVarTypesMap()
    });

    let iteration = 0;
    const maxIterations = 50;

    while (!this._isPrimalOptimal() && iteration < maxIterations) {
      iteration++;
      const enteringVar = this._findPrimalEnteringVariable();
      if (enteringVar === -1) break;

      const leavingVarRowIndex = this._findPrimalLeavingVariableRow(enteringVar);
      if (leavingVarRowIndex === -1) {
        this.steps.push({ type: 'unbounded', message: 'Problème non borné (Primal).' });
        return { status: 'unbounded', steps: this.steps, message: 'Problème non borné (Primal).' };
      }
      
      const leavingVarActual = this.basicVars[leavingVarRowIndex];
      this.pivot(leavingVarRowIndex, enteringVar);
      this.basicVars[leavingVarRowIndex] = enteringVar;

      this.steps.push({
        type: 'iteration', message: `Itération Primal ${iteration}: x${enteringVar + 1} entre, var base ligne ${leavingVarRowIndex + 1} (col ${leavingVarActual+1}) sort.`,
        tableau: this.copyTableau(), entering: enteringVar, leavingRow: leavingVarRowIndex,
        basicVars: [...this.basicVars], numOriginalVars: this.originalVarCount,
        varTypes: this._getVarTypesMap()
      });
    }

    if (iteration >= maxIterations) {
      this.steps.push({ type: 'max_iterations', message: "Max itérations (Primal)." });
      return { status: 'max_iterations', steps: this.steps, message: "Max itérations (Primal)." };
    }

    // Vérifier infaisabilité (variable artificielle en base avec valeur > 0)
    for(let i=0; i < this.basicVars.length; i++){
      const basicVarIndex = this.basicVars[i];
      const rhsVal = this.tableau[i][this.tableau[0].length - 1];
      if(basicVarIndex >= this.originalVarCount + this.numSlackVars + this.numSurplusVars && Math.abs(rhsVal) > this.epsilon){
          this.steps.push({ type: 'infeasible', message: "Problème infaisable (var. artificielle en base avec valeur non nulle)."});
          return { status: 'infeasible', steps: this.steps, message: "Problème infaisable (var. artificielle en base avec valeur non nulle)." };
      }
    }
    
    const result = this.extractSolution();
    this.steps.push({
        type: 'optimal', message: 'Solution optimale (Primal).', solution: result,
        tableau: this.copyTableau(), basicVars: [...this.basicVars],
        numOriginalVars: this.originalVarCount, varTypes: this._getVarTypesMap()
    });
    return { status: 'optimal', solution: result, steps: this.steps };
  }

  _isPrimalOptimal() {
    const lastRow = this.tableau[this.tableau.length - 1];
    for (let j = 0; j < lastRow.length - 1; j++) {
      if (lastRow[j] < -this.epsilon) return false;
    }
    return true;
  }

  _findPrimalEnteringVariable() {
    const lastRow = this.tableau[this.tableau.length - 1];
    let minValue = -this.epsilon; // Cherche le plus négatif
    let enteringVar = -1;
    for (let j = 0; j < lastRow.length - 1; j++) {
      if (lastRow[j] < minValue) {
        minValue = lastRow[j];
        enteringVar = j;
      }
    }
    return enteringVar;
  }

  _findPrimalLeavingVariableRow(enteringVarCol) {
    let minRatio = Infinity;
    let leavingVarRow = -1;
    for (let i = 0; i < this.numConstraints; i++) {
      const pivotCandidate = this.tableau[i][enteringVarCol];
      if (pivotCandidate > this.epsilon) {
        const ratio = this.tableau[i][this.tableau[0].length - 1] / pivotCandidate;
        if (ratio >= 0 && ratio < minRatio) { // Ratio non-négatif
          minRatio = ratio;
          leavingVarRow = i;
        } else if (ratio < 0 && ratio > minRatio - this.epsilon && minRatio === Infinity) {
            // cas où tous les ratios sont négatifs, ce qui indique un problème non borné mais le critère doit être bien géré
        } else if (Math.abs(ratio - minRatio) < this.epsilon && ratio >=0) { // Règle de Bland si ratios égaux
            // Pour simplifier, on prend le premier trouvé. Une implémentation complète utiliserait Bland.
             if (this.basicVars[i] < (leavingVarRow !== -1 ? this.basicVars[leavingVarRow] : Infinity) ) { // exemple de règle de Bland
                leavingVarRow = i;
             }
        }

      }
    }
    return leavingVarRow;
  }


  // --- Méthodes pour le Simplexe Dual ---
  _prepareForDual() {
    // Le `page.js` a déjà transformé les contraintes AX >= b en -AX <= -b.
    // `this.original_A`, `this.original_b`, `this.original_inequalities` reflètent cela.
    // La fonction objectif (min cX) est dans `this.original_c`.
    if (this.optimizationType !== 'minimize') {
      throw new Error("Simplexe dual est typiquement appliqué aux problèmes de minimisation préparés.");
    }

    this.c = [...this.original_c]; // Pour minimisation, c reste c
    this.A = this.original_A.map(row => [...row]); // A et b sont ceux transformés par l'appelant
    this.b = [...this.original_b];
    // Les inégalités passées devraient toutes être '<=' par la logique de `page.js`
    this.inequalities = [...this.original_inequalities]; 

    this.numVars = this.c.length;
    this.numConstraints = this.b.length;
    
    // Pour le dual simple, on n'ajoute que des variables d'écart.
    this.numSlackVars = this.numConstraints;
    this.numSurplusVars = 0;
    this.numArtificialVars = 0;

    // Vérification de la dual-admissibilité (tous c_j >= 0 pour min)
    for (let coeff of this.c) {
      if (coeff < -this.epsilon) {
        // Pour un dual "pur", cela ne devrait pas arriver. On pourrait lancer une erreur
        // ou indiquer que le problème n'est pas directement solvable par le dual simple.
        console.warn("Attention: Démarrage du simplexe dual avec des coefficients de fonction objectif (min) négatifs. Ceci requiert une phase I primale ou une gestion spéciale.");
        // throw new Error("Fonction objectif non dual-admissible (coeff < 0 pour min).");
      }
    }
  }

  _buildTableauForDual() {
    const totalTableauVars = this.numVars + this.numSlackVars; // Pas de surplus/artificiel pour dual simple
    const rows = this.numConstraints + 1;
    const cols = totalTableauVars + 1;

    this.tableau = Array(rows).fill(null).map(() => Array(cols).fill(0));
    this.basicVars = Array(this.numConstraints).fill(-1);

    // Remplir contraintes: A_i X + s_i = b_i (où A et b sont déjà transformés: -A, -b)
    for (let i = 0; i < this.numConstraints; i++) {
      for (let j = 0; j < this.numVars; j++) {
        this.tableau[i][j] = this.A[i][j];
      }
      const slackVarCol = this.numVars + i;
      this.tableau[i][slackVarCol] = 1; // Variable d'écart
      this.basicVars[i] = slackVarCol;
      this.tableau[i][cols - 1] = this.b[i]; // RHS (qui peut être négatif)
    }

    // Ligne Z pour minimisation: c_j pour variables originales, 0 pour slack. Z_val = 0.
    for (let j = 0; j < this.numVars; j++) {
      this.tableau[rows - 1][j] = this.c[j];
    }
    this.tableau[rows - 1][cols - 1] = 0; // Z initial
  }

  _solveDual() {
    this._prepareForDual();
    this._buildTableauForDual();

    this.steps.push({
      type: 'initial_dual', message: 'Tableau initial (Simplexe Dual)',
      tableau: this.copyTableau(), basicVars: [...this.basicVars],
      numOriginalVars: this.originalVarCount, numSlackVars: this.numSlackVars,
      varTypes: this._getVarTypesMap() // Pour aider à nommer les colonnes des slack vars
    });
    
    let iteration = 0;
    const maxIterations = 50;

    while (!this._isDualOptimal() && iteration < maxIterations) {
      iteration++;
      const leavingRow = this._findDualLeavingVariableRow();
      if (leavingRow === -1) { // Normalement géré par _isDualOptimal
         console.log("Dual Optimal mais _isDualOptimal() est false, ceci est étrange."); break;
      }

      const enteringCol = this._findDualEnteringVariableCol(leavingRow);
      if (enteringCol === -1) {
        this.steps.push({ type: 'infeasible_dual', message: "Problème infaisable (Dual) - Le primal est non borné ou infaisable." });
        return { status: 'infeasible_dual', steps: this.steps, message: "Problème infaisable (Dual) - Le primal est non borné ou infaisable." };
      }
      
      const leavingVarActual = this.basicVars[leavingRow];
      this.pivot(leavingRow, enteringCol);
      this.basicVars[leavingRow] = enteringCol;

      this.steps.push({
        type: 'iteration_dual', message: `Itération Dual ${iteration}: Var. base ligne ${leavingRow + 1} (col ${leavingVarActual+1}) sort, x${enteringCol + 1} entre.`,
        tableau: this.copyTableau(), entering: enteringCol, leavingRow: leavingRow,
        basicVars: [...this.basicVars], numOriginalVars: this.originalVarCount,
        varTypes: this._getVarTypesMap()
      });
    }

    if (iteration >= maxIterations) {
      this.steps.push({ type: 'max_iterations_dual', message: "Max itérations (Dual)." });
      return { status: 'max_iterations_dual', steps: this.steps, message: "Max itérations (Dual)." };
    }

    // Si on arrive ici, le dual est optimal (ou une erreur non capturée)
    const result = this.extractSolution();
    this.steps.push({
      type: 'optimal_dual', message: 'Solution optimale (Dual).', solution: result,
      tableau: this.copyTableau(), basicVars: [...this.basicVars],
      numOriginalVars: this.originalVarCount, varTypes: this._getVarTypesMap()
    });
    return { status: 'optimal', solution: result, steps: this.steps };
  }

  _isDualOptimal() { // Tous les RHS >= 0
    for (let i = 0; i < this.numConstraints; i++) {
      if (this.tableau[i][this.tableau[0].length - 1] < -this.epsilon) return false;
    }
    return true;
  }

  _findDualLeavingVariableRow() { // RHS le plus négatif
    let minRHS = -this.epsilon; 
    let leavingRow = -1;
    for (let i = 0; i < this.numConstraints; i++) {
      const rhsVal = this.tableau[i][this.tableau[0].length - 1];
      if (rhsVal < minRHS) {
        minRHS = rhsVal;
        leavingRow = i;
      }
    }
    return leavingRow;
  }

  _findDualEnteringVariableCol(leavingRow) { // Ratio min |Z_j / a_kj| pour a_kj < 0
    let minRatio = Infinity;
    let enteringCol = -1;
    const Z_row = this.tableau[this.tableau.length - 1];
    const pivot_row_coeffs = this.tableau[leavingRow];

    for (let j = 0; j < Z_row.length - 1; j++) { // Exclure RHS
      if (pivot_row_coeffs[j] < -this.epsilon) { // Dénominateur doit être négatif
        // On ne s'attend pas à ce que Z_row[j] soit négatif si dual-admissible initialement
        // mais les pivots peuvent changer cela. Le critère est min { Zj / (-a_kj) } pour a_kj < 0
        // ou min { -Zj / a_kj }. Pour être sûr, on prend la valeur absolue du ratio.
        const ratio = Math.abs(Z_row[j] / pivot_row_coeffs[j]); 
        if (Z_row[j] >= -this.epsilon) { // Zj doit être non-négatif pour minimisation
            if (ratio < minRatio) {
                minRatio = ratio;
                enteringCol = j;
            } else if (Math.abs(ratio - minRatio) < this.epsilon) {
                // Règle de Bland (simplifiée): choisir la variable avec le plus petit indice
                 if (j < enteringCol) {
                    enteringCol = j;
                 }
            }
        }
      }
    }
    return enteringCol;
  }

  // --- Méthodes communes ---
  pivot(pivotRow, pivotCol) {
    const pivotElement = this.tableau[pivotRow][pivotCol];
    if (Math.abs(pivotElement) < this.epsilon) {
      throw new Error(`Élément pivot nul ou proche de zéro (${pivotElement}) à [${pivotRow}, ${pivotCol}]. Impossible de pivoter.`);
    }
    
    for (let j = 0; j < this.tableau[pivotRow].length; j++) {
      this.tableau[pivotRow][j] /= pivotElement;
    }
    // S'assurer que l'élément pivot est bien 1 après normalisation (corriger erreurs flottantes)
    this.tableau[pivotRow][pivotCol] = 1.0; 
    
    for (let i = 0; i < this.tableau.length; i++) {
      if (i !== pivotRow) {
        const factor = this.tableau[i][pivotCol];
        for (let j = 0; j < this.tableau[i].length; j++) {
          this.tableau[i][j] -= factor * this.tableau[pivotRow][j];
        }
        // S'assurer que la colonne pivot (hors ligne pivot) est bien 0 (corriger erreurs flottantes)
        this.tableau[i][pivotCol] = 0.0; 
      }
    }
  }

  extractSolution() {
    const solutionVariables = Array(this.originalVarCount).fill(0);
    for (let i = 0; i < this.numConstraints; i++) {
      const basicVarIndexInTableau = this.basicVars[i];
      if (basicVarIndexInTableau < this.originalVarCount) {
        solutionVariables[basicVarIndexInTableau] = this.tableau[i][this.tableau[0].length - 1];
      }
    }
    
    let objectiveValue = this.tableau[this.tableau.length - 1][this.tableau[0].length - 1];
    if (this.isMax) { // Si c'était max, le tableau a résolu min(-Z), donc Z_val = -(-Z_val)
      objectiveValue = -objectiveValue;
    }
    // Pour min, Z_val est correct.
    
    // Nettoyer les petites valeurs proches de zéro dans la solution
    const cleanedSolutionVariables = solutionVariables.map(val => Math.abs(val) < this.epsilon ? 0 : val);
    if (Math.abs(objectiveValue) < this.epsilon) objectiveValue = 0;

    return {
      variables: cleanedSolutionVariables,
      objectiveValue: objectiveValue
    };
  }
  
  _getVarTypesMap() {
    // Aide à nommer les colonnes pour l'affichage des étapes
    const varTypes = {};
    let sIdx = 1, rIdx = 1, aIdx = 1; // surplus, artificial
    for (let j=0; j < this.numVars + this.numSlackVars + this.numSurplusVars + this.numArtificialVars; j++) {
        if (j < this.numVars) varTypes[j] = `x${j+1}`;
        else if (j < this.numVars + this.numSlackVars) varTypes[j] = `s${sIdx++}`;
        else if (j < this.numVars + this.numSlackVars + this.numSurplusVars) varTypes[j] = `r${rIdx++}`; // r pour surplus (eXces)
        else varTypes[j] = `a${aIdx++}`;
    }
    return varTypes;
  }


  copyTableau() {
    return this.tableau.map(row => [...row]);
  }
}