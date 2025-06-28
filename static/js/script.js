$(document).ready(function() {
    const chatMessages = $('#chat-messages');
    const userInput = $('#user-input');
    const sendBtn = $('#send-btn');
    
    // Handle suggestion button clicks
    $('.suggestion-btn').click(function() {
        const suggestion = $(this).text();
        userInput.val(suggestion);
    });

    // Tombol refresh suggestions
    $('#refresh-suggestions').click(function(e) {
        e.preventDefault();
        loadSuggestions();
    });
    
    // Handle send button click
    sendBtn.click(sendMessage);
    
    // Handle Enter key press
    userInput.keypress(function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });

    // Fungsi untuk memuat suggestion dari backend
    function loadSuggestions() {
        // Tampilkan loading state
        $('#refresh-suggestions').addClass('loading');

        $.ajax({
            url: '/get_suggestions',
            type: 'GET',
            success: function(response) {
                const suggestionsContainer = $('#suggestion-buttons');
                suggestionsContainer.empty(); // Kosongkan container terlebih dahulu
                
                // Tambahkan setiap suggestion sebagai button
                response.suggestions.forEach(suggestion => {
                    const button = $(`<button class="suggestion-btn">${suggestion}</button>`);
                    button.click(function() {
                        // Ketika suggestion diklik, masukkan teks ke input dan kirim
                        $('#user-input').val(suggestion);
                        sendMessage();
                    });
                    suggestionsContainer.append(button);
                });
            },
            error: function(error) {
                console.error('Error loading suggestions:', error);
                // Fallback suggestions jika gagal mengambil dari backend
                const fallbackSuggestions = [
                    "Bagaimana prosedur pelaksanaan persiapan pupuk organik lahan cabai rawit?",
                    "Jelaskan tahapan pengolahan tanah untuk penanaman cabai rawit?",
                    "Bagaimana pengaturan mulsa dan bedengan yang direkomendasikan untuk cabai?",
                    "Bagaimana cara menyediakan benih cabai rawit yang baik?"
                ];
                
                const suggestionsContainer = $('#suggestion-buttons');
                suggestionsContainer.empty();
                
                fallbackSuggestions.forEach(suggestion => {
                    const button = $(`<button class="suggestion-btn">${suggestion}</button>`);
                    button.click(function() {
                        $('#user-input').val(suggestion);
                        sendMessage();
                    });
                    suggestionsContainer.append(button);
                });
            },
            complete: function() {
                // Hilangkan loading state
                $('#refresh-suggestions').removeClass('loading');
            }
        });
    }
    
    function sendMessage() {
        const message = userInput.val().trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessage(message, 'user');
        userInput.val('');
        console.log("user hasil:",{message})
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to backend
        $.ajax({
            url: '/get_response',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(response) {
                // Remove typing indicator
                removeTypingIndicator();
                
                // Add AI response to chat
                addMessage(response.response, 'ai');
                console.log("ai hasil:",{response})
            },
            error: function(xhr, status, error) {
                removeTypingIndicator();
                addMessage("Sorry, I'm having trouble connecting to the AI. Please try again later.", 'ai');
                console.error('Error:', error);
            }
        });
    }
    
    function addMessage(text, sender) {
        const messageClass = sender === 'user' ? 'user-message' : 'ai-message';
        const messageElement = $('<div>').addClass('message ' + messageClass).text(text);
        chatMessages.append(messageElement);
        chatMessages.scrollTop(chatMessages[0].scrollHeight);
    }
    
    function showTypingIndicator() {
        const typingElement = $('<div>').addClass('typing-indicator');
        for (let i = 0; i < 3; i++) {
            typingElement.append($('<div>').addClass('typing-dot'));
        }
        chatMessages.append(typingElement);
        chatMessages.scrollTop(chatMessages[0].scrollHeight);
    }
    
    function removeTypingIndicator() {
        $('.typing-indicator').remove();
    }

    // Muat suggestions saat halaman pertama kali dimuat
    loadSuggestions();
});