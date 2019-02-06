;;; mlir-mode.el --- Major mode for the MLIR assembler language.

;; Maintainer:
;; Version: 0.0

;;; Commentary:

;; Major mode for editing MLIR files.

;;; Code:

(defvar mlir-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?% "_" table)
    (modify-syntax-entry ?@ "_" table)
    (modify-syntax-entry ?# "_" table)
    (modify-syntax-entry ?. "_" table)
    (modify-syntax-entry ?/ ". 12" table)
    (modify-syntax-entry ?\n "> " table)
    table)
  "Syntax table used while in MLIR mode.")

(defvar mlir-font-lock-keywords
  (list
   ;; Variables
   '("%[-a-zA-Z$._0-9]*" . font-lock-variable-name-face)
   ;; Functions
   '("@[-a-zA-Z$._0-9]*" . font-lock-function-name-face)
   ;; Affinemaps
   '("#[-a-zA-Z$._0-9]*" . font-lock-variable-name-face)
   ;; Types
   '("\\b\\(f16\\|bf16\\|f32\\|f64\\|index\\|tf_control\\|i[1-9][0-9]*\\)\\b" . font-lock-type-face)
   '("\\b\\(tensor\\|vector\\|memref\\)\\b" . font-lock-type-face)
   ;; Dimension lists
   '("\\b\\([0-9?]+x\\)*\\(f16\\|bf16\\|f32\\|f64\\|index\\|i[1-9][0-9]*\\)\\b" . font-lock-preprocessor-face)
   ;; Integer literals
   '("\\b[-]?[0-9]+\\b" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("\\b[-+]?[0-9]+.[0-9]*\\([eE][-+]?[0-9]+\\)?\\b" . font-lock-preprocessor-face)
   ;; Hex constants
   '("\\b0x[0-9A-Fa-f]+\\b" . font-lock-preprocessor-face)
   ;; Keywords
   `(,(regexp-opt
       '(;; Toplevel entities
         "br" "ceildiv" "func" "cond_br" "else" "extfunc" "false" "floordiv" "affine.for" "if" "mod" "return" "size" "step" "to" "true" "??" ) 'symbols) . font-lock-keyword-face))
  "Syntax highlighting for MLIR.")

;; Emacs 23 compatibility.
(defalias 'mlir-mode-prog-mode
  (if (fboundp 'prog-mode)
      'prog-mode
    'fundamental-mode))

;;;###autoload
(define-derived-mode mlir-mode mlir-mode-prog-mode "MLIR"
  "Major mode for editing MLIR source files.
\\{mlir-mode-map}
  Runs `mlir-mode-hook' on startup."
  (setq font-lock-defaults `(mlir-font-lock-keywords))
  (setq-local comment-start "//"))

;; Associate .mlir files with mlir-mode
;;;###autoload
(add-to-list 'auto-mode-alist (cons "\\.mlir\\'" 'mlir-mode))

(provide 'mlir-mode)

;;; mlir-mode.el ends here
