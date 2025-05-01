FROM maven:3.8-openjdk-17-slim AS build
COPY . /app
WORKDIR /app
RUN mvn clean install -DskipTests
FROM eclipse-temurin:17-jre-alpine
COPY --from=build /app/target/*.jar /app/my-spring-boot-app.jar
WORKDIR /app
EXPOSE 8030
CMD ["java", "-jar", "my-spring-boot-app.jar"]
