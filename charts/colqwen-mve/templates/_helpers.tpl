{{/*
The served model name is used as the InferenceService and ModelServer name.
*/}}
{{- define "colqwen-mve.modelName" -}}
{{- default "colqwen" .Values.model.name -}}
{{- end -}}

{{- define "colqwen-mve.runtimeName" -}}
{{- default (printf "%s-runtime" (include "colqwen-mve.modelName" .)) .Values.runtime.name -}}
{{- end -}}

{{- define "colqwen-mve.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}
