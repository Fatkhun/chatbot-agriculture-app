document.addEventListener('DOMContentLoaded', function() {
    const chatBody = document.getElementById('chat-body');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const suggestionBtns = document.querySelectorAll('.suggestion-btn');
    
    // Tambahkan pesan sambutan
    addBotMessage("Selamat datang di Konsultasi Pertanian Cabai Rawit. Silakan ajukan pertanyaan Anda tentang budidaya cabai rawit.");
    
    // Fungsi untuk menambahkan pesan pengguna
    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'user-message');
        messageDiv.textContent = message;
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    // Fungsi untuk menambahkan pesan bot
    function addBotMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'bot-message');
        
        // Format jawaban dengan line breaks
        const formattedMessage = message.replace(/\n/g, '<br>');
        messageDiv.innerHTML = formattedMessage;
        
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    // Fungsi untuk menampilkan indikator typing
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot-message', 'typing-indicator');
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
        chatBody.appendChild(typingDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    // Fungsi untuk menghapus indikator typing
    function hideTypingIndicator() {
        const typingDiv = document.getElementById('typing-indicator');
        if (typingDiv) {
            typingDiv.remove();
        }
    }
    
    // Fungsi untuk mengirim pesan
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addUserMessage(message);
            userInput.value = '';
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const data = await response.json();
                hideTypingIndicator();
                addBotMessage(data.response);
            } catch (error) {
                console.error('Error:', error);
                hideTypingIndicator();
                addBotMessage("Maaf, terjadi masalah koneksi. Silakan coba lagi.");
            }
        }
    }
    
    // Event listener untuk tombol kirim
    sendBtn.addEventListener('click', sendMessage);
    
    // Event listener untuk enter
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Event listener untuk tombol saran
    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            userInput.value = this.textContent;
            userInput.focus();
        });
    });
});