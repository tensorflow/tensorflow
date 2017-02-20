/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow;

import java.util.Set;

/**
 * SavedModel representation once the SavedModel is loaded from storage.
 */
public class SavedModelBundle implements AutoCloseable {

    SavedModelBundle(Graph graph, Session session, byte[] metaGraphDef) {
        this.graph = graph;
        this.session = session;
        this.metaGraphDef = metaGraphDef;
    }

    /**
     * Get the metagraphdef associated with the saved model bundle and tags used to load it.
     */
    public byte[] metaGraphDef() {
        return metaGraphDef;
    }

    /**
     * Get the graph loaded from the saved model bundle.
     */
    public Graph graph() {
        return graph;
    }

    /**
     * Get the session associated with the saved model bundle.
     */
    public Session session() {
        return session;
    }

    /**
     * Release resources associated with the saved model bundle.
     *
     * <p>The session and graph are released when this method is called.
     */
    @Override
    public void close() {
        session.close();
        graph.close();
    }

    private final Graph graph;
    private final Session session;
    private final byte[] metaGraphDef;

    /**
     * Load a saved model from an export directory.
     * @param exportDir the directory path containing a saved model.
     * @param tags the tags identifying the specific metagraphdef to load.
     * @return a bundle containing the graph and associated session.
     */
    public static SavedModelBundle loadSavedModel(String exportDir, Set<String> tags) {
        return load(exportDir, tags.toArray(new String[tags.size()]));
    }

    /**
     * Create a SavedModelBundle object from a handle to the C TF_Graph object and
     * to the C TF_Session object, plus the associated metagraphdef.
     *
     * <p>Takes ownership of the handles.
     */
    static SavedModelBundle fromHandle(long graphHandle, long sessionHandle, byte[] metaGraphDef) {
        Graph graph = new Graph(graphHandle);
        Session session = new Session(graph, sessionHandle);
        return new SavedModelBundle(graph, session, metaGraphDef);
    }

    private static native SavedModelBundle load(String exportDir, String[] tags);

    static {
        TensorFlow.init();
    }
}
