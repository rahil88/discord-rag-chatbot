const { Client, GatewayIntentBits, SlashCommandBuilder, REST, Routes, EmbedBuilder } = require('discord.js');
const axios = require('axios');
const logger = require('./logger');
require('dotenv').config();

// Create a new client instance
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages
    ]
});

// Define slash commands
const commands = [
    new SlashCommandBuilder()
        .setName('ping')
        .setDescription('Replies with Pong!'),
    new SlashCommandBuilder()
        .setName('ask')
        .setDescription('Ask a question and get an intelligent response using RAG')
        .addStringOption(option =>
            option.setName('question')
                .setDescription('The question you want to ask')
                .setRequired(true)
        ),
];

// Function to call the RAG API
async function askRAGService(question) {
    const startTime = Date.now();
    const apiUrl = process.env.API_SERVER_URL || 'http://localhost:8000';
    
    logger.info('Calling RAG API', {
        question: question,
        apiUrl: apiUrl,
        timestamp: new Date().toISOString()
    });
    
    try {
        const response = await axios.post(`${apiUrl}/api/ask`, {
            query: question,
            top_k: 5,
            min_similarity: 0.15
        }, {
            timeout: 30000, // 30 second timeout
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const processingTime = Date.now() - startTime;
        
        logger.info('RAG API call successful', {
            question: question,
            processingTime: processingTime,
            responseSize: JSON.stringify(response.data).length,
            sourcesFound: response.data.total_sources_found,
            confidence: response.data.confidence
        });
        
        return response.data;
    } catch (error) {
        const processingTime = Date.now() - startTime;
        
        logger.error('Error calling RAG API', {
            question: question,
            error: error.message,
            errorCode: error.code,
            processingTime: processingTime,
            apiUrl: apiUrl,
            stack: error.stack
        });
        
        throw error;
    }
}

// Function to split text into chunks that fit Discord's embed field limit
function splitTextIntoChunks(text, maxLength = 1000) {
    const chunks = [];
    let currentChunk = '';
    
    // Split by paragraphs first to maintain readability
    const paragraphs = text.split('\n\n');
    
    for (const paragraph of paragraphs) {
        // If adding this paragraph would exceed the limit
        if (currentChunk.length + paragraph.length + 2 > maxLength) {
            // If current chunk has content, save it
            if (currentChunk.trim()) {
                chunks.push(currentChunk.trim());
                currentChunk = '';
            }
            
            // If single paragraph is too long, split it by sentences
            if (paragraph.length > maxLength) {
                const sentences = paragraph.split('. ');
                for (const sentence of sentences) {
                    if (currentChunk.length + sentence.length + 2 > maxLength) {
                        if (currentChunk.trim()) {
                            chunks.push(currentChunk.trim());
                            currentChunk = '';
                        }
                    }
                    currentChunk += (currentChunk ? '. ' : '') + sentence;
                }
            } else {
                currentChunk = paragraph;
            }
        } else {
            currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
        }
    }
    
    // Add the last chunk if it has content
    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }
    
    return chunks;
}

// Function to create an embed for the response
function createResponseEmbed(question, ragResponse, answerChunk = null, chunkIndex = 0, totalChunks = 1) {
    const embed = new EmbedBuilder()
        .setTitle('🤖 RAG Response')
        .setColor(0x00AE86)
        .addFields(
            { name: '❓ Question', value: question, inline: false },
            { name: '💡 Answer', value: answerChunk || ragResponse.answer, inline: false }
        )
        .setFooter({ 
            text: `Confidence: ${(ragResponse.confidence * 100).toFixed(1)}% • Processing time: ${ragResponse.processing_time.toFixed(2)}s${totalChunks > 1 ? ` • Part ${chunkIndex + 1}/${totalChunks}` : ''}` 
        })
        .setTimestamp();

    // Add sources only to the first message
    if (chunkIndex === 0 && ragResponse.sources && ragResponse.sources.length > 0) {
        const sourcesText = ragResponse.sources
            .slice(0, 3) // Limit to first 3 sources to avoid embed limits
            .map((source, index) => 
                `${index + 1}. **${source.source_document}** (Score: ${source.similarity_score.toFixed(3)})\n   ${source.content.substring(0, 200)}${source.content.length > 200 ? '...' : ''}`
            )
            .join('\n\n');
        
        // Ensure sources text doesn't exceed Discord's field limit
        const maxSourcesLength = 1000; // Leave buffer for Discord's 1024 char limit
        let finalSourcesText = sourcesText;
        if (finalSourcesText.length > maxSourcesLength) {
            finalSourcesText = finalSourcesText.substring(0, maxSourcesLength) + '...';
        }
        
        embed.addFields({ 
            name: `📚 Sources (${ragResponse.total_sources_found} found)`, 
            value: finalSourcesText, 
            inline: false 
        });
    }

    return embed;
}

// Register slash commands
async function registerCommands() {
    const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);
    
    try {
        logger.info('Started refreshing application slash commands', {
            commandCount: commands.length,
            clientId: process.env.CLIENT_ID
        });
        
        await rest.put(
            Routes.applicationCommands(process.env.CLIENT_ID),
            { body: commands },
        );
        
        logger.info('Successfully registered application slash commands', {
            commandCount: commands.length,
            commands: commands.map(cmd => cmd.name)
        });
    } catch (error) {
        logger.error('Error registering slash commands', {
            error: error.message,
            errorCode: error.code,
            stack: error.stack
        });
    }
}

// This event will run every time a user interacts with the bot (e.g., uses a slash command)
client.on('interactionCreate', async interaction => {
    // 1. Check if the interaction is a slash command.
    // We do this to ignore other interactions like button clicks for now.
    if (!interaction.isCommand()) return;

    // 2. Destructure the commandName from the interaction object.
    const { commandName } = interaction;

    logger.info('Slash command received', {
        command: commandName,
        userId: interaction.user.id,
        username: interaction.user.tag,
        guildId: interaction.guildId,
        channelId: interaction.channelId
    });

    // 3. Check if the command used was '/ping'.
    if (commandName === 'ping') {
        // 4. If it was, reply with "Pong!".
        // The 'await' is important to ensure the reply is sent properly.
        await interaction.reply('Pong!');
        
        logger.info('Ping command executed', {
            userId: interaction.user.id,
            username: interaction.user.tag
        });
    }
    
    // Handle the '/ask' command
    if (commandName === 'ask') {
        const question = interaction.options.getString('question');
        
        logger.info('Ask command received', {
            question: question,
            userId: interaction.user.id,
            username: interaction.user.tag,
            guildId: interaction.guildId,
            channelId: interaction.channelId
        });
        
        // Defer the reply since this might take a while
        await interaction.deferReply();
        
        try {
            const startTime = Date.now();
            
            // Call the RAG service
            const ragResponse = await askRAGService(question);
            
            // Split the answer into chunks if it's too long
            const answerChunks = splitTextIntoChunks(ragResponse.answer);
            
            // Send the first message
            const firstEmbed = createResponseEmbed(question, ragResponse, answerChunks[0], 0, answerChunks.length);
            await interaction.editReply({ embeds: [firstEmbed] });
            
            // Send additional messages if there are more chunks
            for (let i = 1; i < answerChunks.length; i++) {
                const followUpEmbed = createResponseEmbed(question, ragResponse, answerChunks[i], i, answerChunks.length);
                await interaction.followUp({ embeds: [followUpEmbed] });
                
                // Add a small delay between messages to avoid rate limiting
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
            const totalTime = Date.now() - startTime;
            
            logger.info('Ask command executed successfully', {
                question: question,
                userId: interaction.user.id,
                username: interaction.user.tag,
                totalTime: totalTime,
                messageCount: answerChunks.length,
                sourcesFound: ragResponse.total_sources_found,
                confidence: ragResponse.confidence
            });
            
        } catch (error) {
            logger.error('Error processing ask command', {
                question: question,
                userId: interaction.user.id,
                username: interaction.user.tag,
                error: error.message,
                errorCode: error.code,
                stack: error.stack
            });
            
            // Create error embed
            const errorEmbed = new EmbedBuilder()
                .setTitle('❌ Error')
                .setColor(0xFF0000)
                .setDescription('Sorry, I encountered an error while processing your question. Please try again later.')
                .addFields(
                    { name: 'Question', value: question, inline: false },
                    { name: 'Error', value: error.message || 'Unknown error occurred', inline: false }
                )
                .setTimestamp();
            
            await interaction.editReply({ embeds: [errorEmbed] });
        }
    }
});

// When the client is ready, run this code (only once)
client.once('ready', async () => {
    logger.info('Discord bot is ready', {
        botTag: client.user.tag,
        botId: client.user.id,
        guildCount: client.guilds.cache.size,
        userCount: client.users.cache.size,
        uptime: process.uptime()
    });
    
    // Register slash commands
    await registerCommands();
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Promise Rejection', {
        reason: reason,
        promise: promise,
        stack: reason.stack
    });
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', {
        error: error.message,
        stack: error.stack
    });
    process.exit(1);
});

// Log bot login
logger.info('Starting Discord bot', {
    nodeVersion: process.version,
    platform: process.platform,
    environment: process.env.NODE_ENV || 'development'
});

// Log in to Discord with your client's token
client.login(process.env.DISCORD_TOKEN).catch(error => {
    logger.error('Failed to login to Discord', {
        error: error.message,
        errorCode: error.code,
        stack: error.stack
    });
    process.exit(1);
});
