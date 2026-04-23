Documentación de GitHub
Repositorios/Ramas y fusiones/Administración de ramas protegidas/Acerca de las ramas protegidas
Acerca de las ramas protegidas
Puedes proteger las ramas importantes si configuras las reglas de protección de rama, las cuales definen si los colaboradores pueden borrar o hacer subidas forzadas a la rama y configura los requisitos para cualquier subida a la rama, tal como que pasen las verificaciones de estado o un historial de confirmaciones linear.

¿Quién puede utilizar esta característica?
Las ramas protegidas están disponibles en repositorios públicos con GitHub Free y GitHub Free para las organizaciones. Las ramas protegidas también están disponibles en repositorios públicos y privados con GitHub Pro, GitHub Team, GitHub Enterprise Cloud y GitHub Enterprise Server. Para más información, consulta Planes de GitHub.

En este artículo
Acerca de las reglas de protección de rama
Sugerencia

Si usas reglas de protección de rama que necesiten comprobaciones de estado específicas, asegúrate de que los nombres de trabajo sean únicos en todos los flujos de trabajo. El uso del mismo nombre de trabajo en varios flujos de trabajo puede provocar resultados ambiguos de comprobación de estado y bloquear la combinación de solicitudes de cambios. Consulta Acerca de las verificaciones de estado.

Puedes requerir ciertos flujos de trabajo o requisitos antes de que un colaborador pueda subir los cambios a una rama en tu repositorio, incluyendo la fusión de una solicitud de cambios en la rama, si creas una regla de protección de rama. Los actores solo se pueden agregar para omitir listas cuando el repositorio pertenece a una organización.

Predeterminadamente, cada regla de protección de rama inhabilita las subidas forzadas en las ramas coincidentes y previene que éstas se borren. Opcionalmente, puedes inhabilitar estas restricciones y habilitar la configuración adicional de protección de ramas.

De forma predeterminada, las restricciones de una regla de protección de rama no se aplican a los usuarios con permisos de administrador para el repositorio o roles personalizados con el permiso "omitir protecciones de rama". Opcionalmente, también puedes aplicar las restricciones a administradores y roles con el permiso "omitir protecciones de rama". Para más información, consulta Administrar roles de repositorio personalizados en una organización.

Puede crear una regla de protección de rama en un repositorio para una rama específica, todas las ramas o cualquier rama que coincida con un patrón de nombre que especifique con la sintaxis fnmatch. Por ejemplo, para proteger todas las ramas que contengan la palabra release, puede crear una regla de rama para *release*. Para más información sobre los patrones de nombre de rama, consulta Administrar una regla de protección de rama.

Puedes configurar una solicitud de cambios para que se fusione automáticamente cuando se cumplan todos los requisitos de fusión. Para más información, consulta Fusionar una solicitud de cambios automáticamente.

Nota:

Solo se puede aplicar una regla de protección de una sola rama a la vez, lo que significa que puede ser difícil saber qué regla se aplicará cuando varias versiones de una regla se dirijan a la misma rama. Para información sobre una alternativa a las reglas de protección de ramas, consulta Acerca de los conjuntos de reglas.

Acerca de la configuración de protección de rama
Para cada regla de protección de rama, puedes elegir habilitar o inhabilitar la siguiente configuración.

Revisiones de solicitudes de incorporación de cambios obligatorias antes de la combinación
Requerir verificaciones de estado antes de las fusiones
Requerir la resolución de conversaciones antes de fusionar
Confirmaciones firmadas obligatorias
Requerir historial linear
Requerir cola de fusión
Implementaciones correctas obligatorias antes de la combinación
Bloquear de una rama
No permitir la omisión de la configuración anterior
Restricción de quiénes pueden realizar inserciones en ramas coincidentes
Inserciones forzadas permitidas
Eliminaciones permitidas
Para más información sobre cómo configurar la protección de ramas, consulta Administrar una regla de protección de rama.

Requerir revisiones de solicitudes de cambio antes de fusionarlas
Los administradores de repositorios o roles personalizados con el permiso "editar reglas de repositorio" pueden requerir que todas las solicitudes de incorporación de cambios reciban un número específico de revisiones de aprobación antes de que alguien combine la solicitud de incorporación de cambios con una rama protegida. Puedes requerir revisiones de aprobación de personas con permisos de escritura en el repositorio o de un propietario de código designado.

Si habilitas las revisiones requeridas, los colaboradores solo podrán subir los cambios a una rama protegida a través de una solicitud de cambios que se encuentre aprobada por el total de revisores requeridos con permisos de escritura.

Si un usuario con permisos administrativos elige la opción Solicitar cambios en una revisión, tendrá que aprobar la solicitud de incorporación de cambios antes de que se pueda combinar. Si un revisor que solicita cambios en una solicitud de cambios no está disponible, cualquiera con permisos de escritura para el repositorio podrá descartar la revisión que está haciendo el bloqueo.

Incluso después de que todos los revisores hayan aprobado una solicitud de cambios, los colaboradores no pueden fusionar la solicitud de cambios si hay otra solicitud de cambios abierta que tenga una rama de encabezado que apunte a la misma confirmación y tenga revisiones rechazadas o pendientes. Alguien con permisos de escritura debe aprobar o descartar primero la revisión que está causando el bloqueo en el resto de las solicitudes de cambios.

Si un colaborador intenta fusionar una solicitud de cambios con revisiones rechazadas o pendientes en la rama protegida, el colaborador recibirá un mensaje de error.

remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: error: Changes have been requested.
También puedes optar por descartar aprobaciones de solicitud de cambios obsoletas cuando se insertan confirmaciones que afectan la diferencia en la solicitud de cambios. GitHub registra el estado de la diferencia en el momento en que se aprueba una solicitud de cambios. Este estado representa el conjunto de cambios que el revisor aprobó. Si el estado de la diferencia cambia (por ejemplo, porque un colaborador inserta cambios nuevos en la rama de solicitud de cambios o hace clic en Actualizar rama, o bien porque se combina una solicitud de cambios en la rama de destino), la revisión de aprobación se descarta como obsoleta y la solicitud de cambios no se puede combinar hasta que alguien apruebe nuevamente el trabajo. Para obtener información sobre la rama base, consulta Acerca de las solicitudes de incorporación de cambios.

Opcionalmente, puedes restringir la capacidad para descartar las revisiones de las solicitudes de cambio para que solo puedan hacerlas algunos equipos o personas específicos. Para más información, consulta Descartar una revisión de solicitud de extracción.

Opcionalmente, puedes elegir el requerir revisiones de los propietarios del código. Si lo haces, el propietario de código deberá aprobar cualquier solicitud de cambios que afecte dicho código antes de que la solicitud de cambios pueda fusionarse en la rama protegida.

Opcionalmente, puedes exigir que la inserción revisable más reciente la apruebe alguien que no sea quien la haya insertado. Esto significa que al menos otro revisor autorizado aprobó los cambios. Por ejemplo, el "último revisor" puede comprobar que el conjunto más reciente de cambios incorpora comentarios de otras revisiones y no agrega contenido nuevo no revisado.

En el caso de solicitudes de cambios complejas que requieren muchas revisiones, exigir la aprobación de alguien que no sea la última persona en realizar la inserción puede ser un equilibrio que evite la necesidad de descartar todas las revisiones obsoletas: con esta opción, las revisiones "obsoletas" no se descartan y la solicitud de cambios permanece aprobada siempre y cuando la apruebe alguien que no sea quien hizo los cambios más recientes. A los usuarios que ya han revisado una solicitud de incorporación de cambios se les puede volver a aprobar después de la inserción más reciente para cumplir este requisito. Si le preocupa que "se secuestren" las solicitudes de cambios (donde el contenido no aprobado se agrega a las solicitudes de cambios aprobadas), es más seguro descartar las revisiones obsoletas.

Nota:

Si seleccionas Descartar aprobaciones de solicitudes de incorporación de cambios obsoletas cuando se insertan nuevas confirmaciones o Requerir aprobación de la inserción revisable más reciente, se producirá un error al crear manualmente la confirmación de combinación para una solicitud de incorporación de cambios e insertarla directamente en una rama protegida, a menos que el contenido de la combinación coincida exactamente con la combinación generada por GitHub para la solicitud de incorporación de cambios.

Además, con esta configuración, las revisiones aprobadas se descartarán por obsoletas si la base de combinación introduce nuevos cambios después de enviar la revisión. La base de combinación es la confirmación que es el último antecesor común entre la rama de tema y la rama base. Si cambia la base de combinación, la solicitud de incorporación de cambios no se puede combinar hasta que alguien apruebe el trabajo de nuevo.

Requerir verificaciones de estado antes de las fusiones
Las comprobaciones de estado necesarias deben tener un estado successful, skipped o neutral antes de que los colaboradores puedan realizar cambios en una rama protegida. Las comprobaciones de estado necesarias pueden ser comprobaciones o estados de confirmación. Para más información, consulta Acerca de las verificaciones de estado.

Puede usar la API de estado de confirmación para permitir que los servicios externos marquen las confirmaciones con un estado adecuado. Para más información, consulta Puntos de conexión de la API de REST para estados de confirmaciones.

Después de habilitar las verificaciones de estado requierdas, cualquier verificación de estado deberá pasar antes de que los colaboradores puedan fusionar los cambios en la rama protegida. Una vez que hayan pasado todas las verificaciones de estado requeridas, cualquier confirmación deberá ya sea subirse en otra rama y después fusionarse, o subirse directo a la rama protegida.

Cualquier persona o integración con permisos de escritura en un repositorio puede configurar el estado de cualquier verificación de estado en el repositorio, pero en algunos casos, podrías querer que solo se acepte una verificación de estado desde una GitHub App específica. Cuando agregas una verificación de estado requerida, puedes seleccionar una app que haya configurado esta verificación recientemente como la fuente de actualizaciones de estado esperada. Si otra persona o integración configura el estado, no se podrá hacer la fusión. Si seleccionas "cualquier fuente", aún puedes verificar el autor de cada estado listado en la caja de fusión manualmente.

Puedes configurar las verificaciones de estado requeridas para que sean "laxas" o "estrictas". El tipo de verificación de estado requerida que elijas determina si se requiere que tu rama esté actualizada con la rama base antes de la fusión.

Tipo de verificación de estado requerida	Configuración	Requisitos de fusión	Consideraciones
Strict	La casilla Requerir que las ramas estén actualizadas antes de la combinación está activada.	La rama debe estar actualizada con la rama base antes de la combinación.	Este es el comportamiento predeterminado para las verificaciones de estado requeridas. Se pueden requerir más compilaciones, ya que deberás actualizar la rama principal después de que otros colaboradores actualicen la rama de destino.
Flexible	La casilla Requerir que las ramas estén actualizadas antes de la combinación no está activada.	La rama no tiene que estar actualizada con la rama base antes de la combinación.	Tendrás menos construcciones requeridas, ya que no necesitarás actualizar la rama de encabezado después de que otros colaboradores fusionen las solicitudes de extracción. Las verificaciones de estado pueden fallar después de que fusiones tu rama si hay cambios incompatibles con la rama de base.
Deshabilitada	La casilla Requerir que se superen las comprobaciones de estado antes de la combinación no está activada.	La rama no tiene restricciones de fusión.	Si las verificaciones de estado requeridas no están habilitadas, los colaboradores pueden fusionar la rama en cualquier momento, independientemente de si está actualizada con la rama de base. Esto aumenta la posibilidad de cambios incompatibles.
Para consultar información sobre solución de problemas, consulta Solución de problemas para verificaciones de estado requeridas.

Requerir la resolución de conversaciones antes de fusionar
Requiere que se resuelvan todos los comentarios de la solicitud de cambios antes de qeu se pueda fusionar con una rama protegida. Esto garantiza que todos los comentarios se traten o reconozcan antes de fusionar.

Requerir confirmaciones firmadas
Al habilitar la firma de confirmación obligatoria en una rama, los colaboradores y bots solo pueden insertar confirmaciones que se hayan firmado y comprobado en la rama. Para más información, consulta Acerca de la verificación de firma de confirmación.

Nota:

Si habilitaste el modo vigilante, el cual indica que tus confirmaciones siempre se firmarán, cualquier confirmación que GitHub identifique como "Verificada parcialmente" se permitirá en aquellas ramas que requieran confirmaciones firmadas. Para obtener más información sobre el modo vigilante, consulta Mostrar los estados de verificación para todas tus confirmaciones.
Si un colaborador sube una confirmación sin firmar a una rama que requiere firmas de confirmación, este necesitará rebasar dicha confirmación para incluir una firma verificada y luego subir forzadamente la confirmación reescrita a esta.
Siempre puede subir confirmaciones locales a la rama si estas se firmaron y verificaron. También puedes combinar confirmaciones firmadas y comprobadas en la rama mediante una solicitud de cambios. Pero no puedes fusionar mediante combinación con "squash" y combinar una solicitud de cambios en la rama en GitHub a menos de que seas el creador de esa solicitud. Puedes fusionar mediante combinación con "squash" y combinar las solicitudes de cambios localmente. Para más información, consulta Revisar solicitudes de extracción localmente.

Para obtener más información sobre los métodos de fusión, consulta Acerca de los métodos de fusión en GitHub.

Requerir un historial linear
El requerir un historial de confirmaciones linear previene que los colaboradores suban confirmaciones de fusión a la rama. Esto significa que cualquier solicitud de extracción fusionada con la rama protegida deberá utilizar una fusión combinada o una fusión de rebase. Un historial de confirmaciones estrictamente linear puede ayudar a los equipos a revertir los cambios con mayor facilidad. Para obtener más información sobre los métodos de fusión, consulta Acerca de las fusiones de las solicitudes de extracción.

Antes de poder requerir un historial de confirmaciones linear, tu repositorio deberá permitir fusiones combinadas o fusiones de rebase. Para más información, consulta Configurar fusiones de solicitudes de extracción.

Requerir una cola de fusión
Una cola de combinación ayuda a aumentar la velocidad mediante la automatización de las combinaciones de solicitudes de incorporación de cambios en una rama ocupada y asegurarse de que la rama nunca se interrumpe por cambios incompatibles.

La cola de combinación proporciona las mismas ventajas que requerir que las ramas estén actualizadas antes de combinar la protección de la rama, pero no requiere que un autor de la solicitud de incorporación de cambios actualice su rama de solicitud de incorporación de cambios y espere a que finalicen las comprobaciones de estado antes de intentar combinar.

El uso de una cola de combinación es especialmente útil en las ramas que tienen un número relativamente alto de solicitudes de incorporación de cambios que se combinan cada día de muchos usuarios diferentes.

Cuando una solicitud de incorporación de cambios ha superado todas las comprobaciones de protección de rama necesarias, el usuario con acceso de escritura al repositorio puede agregar la solicitud de incorporación de cambios a la cola. La cola de combinación garantizará que los cambios de la solicitud de incorporación de cambios pasen todas las comprobaciones de estado necesarias cuando se aplican a la versión más reciente de la rama de destino y a las solicitudes de incorporación de cambios que ya estén en la cola.

Una cola de combinación puede usar las GitHub Actions o tu propio proveedor de CI para ejecutar comprobaciones necesarias en las solicitudes de incorporación de cambios en una cola de combinación. Para más información, consulta Documentación de GitHub Actions.

GitHub combina la solicitud de incorporación de cambios según la estrategia de combinación configurada en la protección de la rama una vez que se superan todas las comprobaciones de CI necesarias. Para más información sobre los métodos de fusión, consulta Administración de una cola de fusión mediante combinación.

Implementaciones correctas obligatorias antes de la combinación
Puede exigir que los cambios se implementen correctamente en entornos específicos antes de poder combinar una rama. Por ejemplo, puede usar esta regla para asegurarse de que los cambios se implementan correctamente en un entorno de ensayo antes de que se combinen en la rama predeterminada.

Bloqueo de una rama
El bloqueo de una rama hará que sea de solo lectura y garantizará que no se pueda realizar ninguna confirmación en ella. Tampoco se pueden eliminar ramas bloqueadas.

De manera predeterminada, un repositorio bifurcado no admite la sincronización desde su repositorio ascendente. Puedes habilitar Permitir la sincronización de bifurcación para extraer los cambios del repositorio ascendente, a la vez que impides otras contribuciones a la rama de la bifurcación.

No permitir la omisión de la configuración anterior
De forma predeterminada, las restricciones de una regla de protección de rama no se aplican a los usuarios con permisos de administrador para el repositorio o roles personalizados con el permiso "omitir protecciones de rama" en un repositorio.

También puedes habilitar esta configuración para aplicar las restricciones a administradores y roles con el permiso "omitir protecciones de rama". Para más información, consulta Administrar roles de repositorio personalizados en una organización.

Restringir quiénes pueden subir a las ramas coincidentes
Puedes habilitar restricciones de rama en repositorios públicos que pertenecen a una organización de GitHub Free y en todos los repositorios que pertenecen a una organización con GitHub Team o GitHub Enterprise Cloud.

Cuando habilitas las restricciones de rama, solo los usuarios, equipos o apps a los que se les haya dado permisos pueden subir información a la rama protegida. Puedes ver y editar los usuarios, equipos o apps con acceso de escritura a una rama protegida en la configuración de la misma. Cuando se requieren las comprobaciones de estado, aún se prevendrá que las personas, equipos y aplicaciones que tienen permiso de subida a una rama protegida realicen fusiones mediante combinación en caso de que fallen las comprobaciones requeridas. Las personas, equipos y apps que tengan permiso de subida a una rama protegida aún necesitarán crear solicitudes de cambio cuando estas se requieran.

Opcionalmente, puedes aplicar las mismas restricciones a la creación de ramas que coincidan con la regla. Por ejemplo, si creas una regla que solo permite que un equipo determinado suba información a cualquier rama que contenga la palabra release, solo los miembros de ese equipo podrían crear una nueva rama con la palabra release.

Solo puedes dar acceso de escritura a una rama protegida, o bien conceder permiso para crear una rama coincidente, a usuarios, equipos o GitHub Apps instaladas con acceso de escritura a un repositorio. Las personas y aplicaciones con permisos administrativos en un repositorio siempre pueden subir información a una rama protegida o crear una rama coincidente.

Permitir las subidas forzadas
De manera predeterminada, GitHub bloquea las inserciones forzadas en todas las ramas protegidas. Cuando habilitas las subidas forzadas a una rama protegida, puedes elegir uno de dos grupos que pueden hacerlas:

Permitir que todos los que por lo menos tengan permisos de escritura en el repositorio suban información forzadamente a la rama, incluyendo aquellos con permisos administrativos.
Permitir que solo personas o equipos específicos suban información forzadamente a la rama.
Si alguien fuerza las inserciones en una rama, la inserción forzada puede significar confirmaciones en las que otros colaboradores basaron su trabajo se quitan del historial de la rama. Las personas pueden tener conflictos de fusión o solicitudes de cambios corruptas. La inserción forzada también se puede usar para eliminar ramas o apuntar una rama a confirmaciones que no se aprobaron en una solicitud de cambios.

Habilitar las subidas forzadas no invalidará ninguna otra regla de protección a la rama. Por ejemplo, si una rama requiere un historial de confirmaciones linear, no puedes forzar la subida de fusión de confirmaciones en esa rama.

Permitir el borrado
Por defecto, no puedes eliminar una rama protegida. Cuando habilitas el borrado de una rama protegida, cualquiera que tenga por lo menos permiso de escritura en el repositorio podrá borrar la rama.

Nota:

Si la rama está bloqueada, no puedes eliminarla aunque tengas permiso para ello.
