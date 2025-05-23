document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const messagesContainer = document.getElementById('messages');
    const suggestionsContainer = document.getElementById('suggestions');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const modelSelector = document.getElementById('model-selector');
    const clearChatButton = document.getElementById('clear-chat');
    
    // State
    let isLoading = false;
    
    // Suggestions - you can either hardcode them or load from server
    const defaultSuggestions = [
        "How to build a convolutional neural network?",
        "How do I compile and train a Keras model?",
        "What’s the use of callbacks like ModelCheckpoint and EarlyStopping?",
        "How do I use tf.data.Dataset to load and preprocess large datasets?"
    ];
    
    // Initial setup
    // Check if history was pre-loaded from Flask
    if (typeof initialChatHistory !== 'undefined' && initialChatHistory.length) {
        renderChatHistory(initialChatHistory);
    } else {
        loadChatHistory();
    }
    
    // Check if suggestions were pre-loaded from Flask
    if (typeof initialSuggestions !== 'undefined' && initialSuggestions.length) {
        renderSuggestions(initialSuggestions);
    } else {
        loadSuggestions();
    }
    
    // Event listeners
    messageInput.addEventListener('input', function() {
        sendButton.disabled = !messageInput.value.trim() || isLoading;
    });
    
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && messageInput.value.trim() && !isLoading) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendButton.addEventListener('click', function() {
        if (messageInput.value.trim() && !isLoading) {
            sendMessage();
        }
    });
    
    modelSelector.addEventListener('change', function() {
        changeModel(modelSelector.value);
    });
    
    clearChatButton.addEventListener('click', function() {
        clearChatHistory();
    });
    
    // Functions
    function loadSuggestions() {
        fetch('/get_suggestions')
            .then(response => response.json())
            .then(data => {
                const suggestions = data.suggestions || defaultSuggestions;
                renderSuggestions(suggestions);
            })
            .catch(error => {
                console.error('Error loading suggestions:', error);
                renderSuggestions(defaultSuggestions);
            });
    }
    
    function renderSuggestions(suggestions) {
        suggestionsContainer.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const button = document.createElement('button');
            button.className = 'suggestion-button';
            button.textContent = suggestion;
            button.addEventListener('click', function() {
                handleSuggestionClick(suggestion);
            });
            
            suggestionsContainer.appendChild(button);
        });
        
        // Only show suggestions if no messages exist
        checkMessagesExist();
    }
    
    function handleSuggestionClick(suggestion) {
        addMessage(suggestion, 'user');
        suggestionsContainer.style.display = 'none';
        sendMessageToServer(suggestion);
    }
    
    function loadChatHistory() {
        fetch('/get_chat_history')
            .then(response => response.json())
            .then(data => {
                if (data.history && Array.isArray(data.history)) {
                    renderChatHistory(data.history);
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
            });
    }
    
    function renderChatHistory(historyArray) {
        historyArray.forEach((message, index) => {
            const sender = index % 2 === 0 ? 'user' : 'assistant';
            addMessage(message, sender);
        });
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();
        addMessage(message, 'user');
        messageInput.value = '';
        sendButton.disabled = true;
        suggestionsContainer.style.display = 'none';
        
        sendMessageToServer(message);
    }
    
    function sendMessageToServer(message) {
        setLoading(true);
        
        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addMessage(data.response, 'assistant');
            } else {
                addMessage('Sorry, I encountered an error processing your request.', 'assistant');
                console.error('Error from server:', data.message);
            }
        })
        .catch(error => {
            console.error('Error sending message:', error);
            addMessage('Network error. Please try again later.', 'assistant');
        })
        .finally(() => {
            setLoading(false);
        });
    }
    
    // Configure marked.js for proper markdown rendering
    marked.setOptions({
        breaks: true,          // Convert \n to <br>
        gfm: true,             // GitHub Flavored Markdown
        headerIds: false,      // Don't add IDs to headers
        mangle: false,         // Don't mangle email links
        sanitize: false,       // Don't sanitize HTML
        smartLists: true,      // Use smarter list behavior
        smartypants: true,     // Use smart typography
        xhtml: false           // Don't close tags with />
    });

    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}`;
        
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble markdown-content';
        
        // Parse markdown content
        messageBubble.innerHTML = marked.parse(text);
        
        // Apply syntax highlighting to code blocks
        const codeBlocks = messageBubble.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.parentNode.classList.add('code-block');
        });
        
        messageElement.appendChild(messageBubble);
        messagesContainer.appendChild(messageElement);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Check if we should show suggestions
        checkMessagesExist();
    }
    
    function checkMessagesExist() {
        const hasMessages = messagesContainer.querySelector('.message') !== null;
        suggestionsContainer.style.display = hasMessages ? 'none' : 'grid';
    }
    
    // Sample document data - in a real app, this would come from your backend
    const sampleDocuments = [
        { id: 1, title: 'Review the conceptual documentation', description: 'Understand the purpose and functionality of RecursiveUrlLoader.' },
        { id: 2, title: 'Check the integration docs', description: 'How-to guides for examples or usage patterns of RecursiveUrlLoader to load content from a web page.' },
        { id: 3, title: 'Summarize the typical code snippet', description: 'Method to instantiate and use RecursiveUrlLoader for loading content recursively from a URL.' }
    ];
    
    // Processing state variables
    let processingProgress = 0;
    let progressInterval;
    let selectedDocuments = [];
    const processingFeedback = document.getElementById('processing-feedback');
    const progressBar = document.getElementById('progress-bar');
    const documentSelection = document.getElementById('document-selection');
    
    function setLoading(loading) {
        isLoading = loading;
        sendButton.disabled = loading || !messageInput.value.trim();
        
        if (loading) {
            // Show processing feedback with progress bar and document selection
            startProcessingFeedback();
        } else {
            // Hide processing feedback
            stopProcessingFeedback();
            
            // Remove simple loading indicator if it exists
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) {
                loadingMessage.remove();
            }
        }
    }
    
    function startProcessingFeedback() {
        // Reset progress
        processingProgress = 0;
        progressBar.style.width = '0%';
        selectedDocuments = [];
        documentSelection.innerHTML = '';
        
        // Show processing feedback
        processingFeedback.style.display = 'block';
        
        // Simulate progress updates
        progressInterval = setInterval(() => {
            // Increment progress (slower at the beginning, faster towards the end)
            if (processingProgress < 30) {
                processingProgress += 1;
            } else if (processingProgress < 60) {
                processingProgress += 2;
            } else if (processingProgress < 85) {
                processingProgress += 1;
            } else if (processingProgress < 95) {
                processingProgress += 0.5;
            }
            
            // Cap at 95% - the last 5% will be filled when the response is received
            if (processingProgress > 95) {
                processingProgress = 95;
                clearInterval(progressInterval);
            }
            
            // Update progress bar
            progressBar.style.width = `${processingProgress}%`;
            
            // Add documents at specific progress points
            if (processingProgress === 15 && selectedDocuments.length === 0) {
                addDocumentCard(sampleDocuments[0]);
            } else if (processingProgress === 40 && selectedDocuments.length === 1) {
                addDocumentCard(sampleDocuments[1]);
            } else if (processingProgress === 70 && selectedDocuments.length === 2) {
                addDocumentCard(sampleDocuments[2]);
            }
        }, 100);
        
        // Scroll to show the processing feedback
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function stopProcessingFeedback() {
        // Clear progress interval
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        // Complete progress to 100%
        progressBar.style.width = '100%';
        
        // Hide processing feedback after a short delay
        setTimeout(() => {
            processingFeedback.style.display = 'none';
        }, 500);
    }
    
    function addDocumentCard(document) {
        // Add to selected documents
        selectedDocuments.push(document);
        
        // Create document card
        const card = document.createElement('div');
        card.className = 'document-card';
        card.dataset.id = document.id;
        
        const title = document.createElement('div');
        title.className = 'document-card-title';
        title.textContent = document.title;
        
        const description = document.createElement('div');
        description.className = 'document-card-description';
        description.textContent = document.description;
        
        const icons = document.createElement('div');
        icons.className = 'document-card-icons';
        
        // Add some document icons (A, B, C, etc.)
        for (let i = 0; i < 3; i++) {
            const icon = document.createElement('div');
            icon.className = 'document-icon';
            icon.textContent = String.fromCharCode(65 + i); // A, B, C, etc.
            icons.appendChild(icon);
        }
        
        // Assemble card
        card.appendChild(title);
        card.appendChild(description);
        card.appendChild(icons);
        
        // Add to document selection
        documentSelection.appendChild(card);
        
        // Scroll to show the new document
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function changeModel(modelId) {
        const formData = new FormData();
        formData.append('model', modelId);
        
        fetch('/select_model', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'success') {
                console.error('Error changing model:', data.message);
            }
        })
        .catch(error => {
            console.error('Error changing model:', error);
        });
    }
    
    function clearChatHistory() {
        fetch('/clear_chat_history', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                messagesContainer.innerHTML = '';
                suggestionsContainer.style.display = 'grid';
            }
        })
        .catch(error => {
            console.error('Error clearing chat history:', error);
        });
    }
});