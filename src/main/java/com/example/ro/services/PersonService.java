package com.example.ro.services;

import com.example.ro.dto.responseDTO.ApiError;
import com.example.ro.dto.requestDTO.PersonDTO;
import com.example.ro.dto.requestDTO.UpdatePersonDTO;
import com.example.ro.models.Person;
import com.example.ro.repositories.PersonRepository;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@AllArgsConstructor
public class PersonService {

    private final PersonRepository personRepository;

    public ApiError getPersonById(int id) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            apiError.setData(optionalPerson.get());
            apiError.setText("user get succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setData(null);
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;

    }

    public ApiError createPerson(PersonDTO dto) {
        ApiError apiError = new ApiError();

        Person person = new Person();

        person.setFirstName(dto.getFirstName());
        person.setFirstName(dto.getFirstName());
        person.setBirthPlace(dto.getBirthPlace());
        person.setGender(dto.getGender());
        person.setRole(dto.getRole());

        Person savedPerson = personRepository.save(person);
        apiError.setValue("201");
        apiError.setText("user created successfully");
        apiError.setData(savedPerson);
        return apiError;
    }

    public ApiError updatePerson(int id, UpdatePersonDTO dto) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            Person person = optionalPerson.get();
            if (dto.getFirstName() != null) {
                person.setFirstName(dto.getFirstName());
            }
            if (dto.getRole() != null){
                person.setRole(dto.getRole());
            }
            if (dto.getGender() != null) {
                person.setGender(dto.getGender());
            }
            if (dto.getLastName() != null) {
                person.setLastName(dto.getLastName());
            }
            if( dto.getBirthPlace() != null) {
                person.setBirthPlace(dto.getBirthPlace());
            }

            Person savedperson = personRepository.save(person);

            apiError.setData(savedperson);
            apiError.setText("user updated succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setData(null);
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;
    }

    public ApiError deletePerson(int id) {
        ApiError apiError = new ApiError();
        Optional<Person> optionalPerson = personRepository.findById(id);
        if (optionalPerson.isPresent()) {
            personRepository.deleteById(id);
            apiError.setData(null);
            apiError.setText("user deleted succesfully");
            apiError.setValue("200");
            return apiError;
        }
        apiError.setData(null);
        apiError.setText("not found");
        apiError.setValue("404");
        return apiError;
    }
}
