{{/*
The served model name. This value is used both as the InferenceService name and
as SERVED_MODEL_NAME inside the container, so KServe routes
/v2/models/<name>/infer to the model registered by ModelServer.
*/}}
{{- define "colqwen-mve.modelName" -}}
{{- default "mxbai" .Values.model.name -}}
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
